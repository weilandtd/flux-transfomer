"""Weighted bipartite graphs from a connectivity matrix: pruning, relabeling, common subgraphs and a
satellite drawing where every target sits next to the sources it connects to most strongly."""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

SOURCE, TARGET = "source", "target"


def bipartite_graph(weights, sources, targets, threshold=0.9, relative=True):
    """Graph of the edges that are at least ``threshold`` of their target's strongest edge, or
    with ``relative=False`` at least ``threshold`` itself.

    Args:
        weights: non-negative array (n_targets x n_sources); row i connects target i to the sources.
        sources: source labels, one per column.
        targets: target labels, one per row.
        threshold: cut in [0, 1] on each row divided by its maximum, and a zero weight is
            never an edge.
        relative: ``False`` keeps the weights as given and cuts them all at one ``threshold``.

    Returns:
        ``nx.Graph`` with nodes ``(side, label)`` carrying ``side`` and ``label`` attributes and
        edges carrying the ``weight`` (relative to its row's maximum unless ``relative=False``).
        Nodes left without an edge are dropped.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(targets), len(sources)):
        raise ValueError(f"weights {weights.shape} != (n_targets, n_sources) "
                         f"({len(targets)}, {len(sources)})")
    if relative:
        peak = weights.max(1, keepdims=True)
        weights = np.divide(weights, peak, out=np.zeros_like(weights), where=peak > 0)
    graph = nx.Graph()
    for i, j in zip(*np.nonzero((weights >= threshold) & (weights > 0))):
        _add_edge(graph, (SOURCE, sources[j]), (TARGET, targets[i]), weights[i, j])
    return graph


def _add_edge(graph, source, target, weight):
    for node in (source, target):
        graph.add_node(node, side=node[0], label=node[1])
    graph.add_edge(source, target, weight=weight)


def _mapper(mapping):
    if mapping is None:
        return lambda label: label
    return mapping if callable(mapping) else mapping.get


def relabel(graph, source_map=None, target_map=None):
    """Map nodes into a shared name space; merged edges keep their largest weight.

    Args:
        graph: graph from :func:`bipartite_graph`.
        source_map, target_map: dict or callable from label to new label; ``None`` keeps the
            labels, and a label mapped to ``None`` (or missing from a dict) drops the node.
    """
    maps = {SOURCE: _mapper(source_map), TARGET: _mapper(target_map)}
    out = nx.Graph()
    for u, v, w in graph.edges(data="weight"):
        source, target = (u, v) if u[0] == SOURCE else (v, u)
        s, t = maps[SOURCE](source[1]), maps[TARGET](target[1])
        if s is None or t is None:
            continue
        if out.has_edge((SOURCE, s), (TARGET, t)):
            w = max(w, out.edges[(SOURCE, s), (TARGET, t)]["weight"])
        _add_edge(out, (SOURCE, s), (TARGET, t), w)
    return out


def common_subgraph(graphs):
    """Edges present in every graph, weighted by their smallest weight.

    Each edge also keeps ``weights``, its weight in every graph in order.
    """
    shared = set.intersection(*(set(map(frozenset, g.edges)) for g in graphs))
    out = nx.Graph()
    for edge in shared:
        source, target = sorted(edge)                      # "source" < "target"
        weights = [g.edges[source, target]["weight"] for g in graphs]
        _add_edge(out, source, target, min(weights))
        out.edges[source, target]["weights"] = weights
    return out


def hubs(graph, side=SOURCE, n=8, min_strength=None):
    """Nodes on ``side`` by weighted degree, strongest first: the ``n`` strongest, or with
    ``min_strength`` every node reaching it (``n=None`` then imposes no count)."""
    strength = graph.degree(weight="weight")
    nodes = [node for node in graph if node[0] == side
             and (min_strength is None or strength[node] >= min_strength)]
    return sorted(nodes, key=lambda node: (-strength[node], node[1]))[:n]


LAYOUTS = ("spring", "biplot", "columns")
ANNOTATE = ("nodes", "legend")


def _sides(graph):
    return ([n for n in graph if n[0] == SOURCE], [n for n in graph if n[0] == TARGET])


def _biplot(graph):
    """Correspondence analysis of the weighted biadjacency: both sides in one plane, a source and
    a target close when they share more weight than their totals predict."""
    sources, targets = _sides(graph)
    A = nx.bipartite.biadjacency_matrix(graph, sources, targets, weight="weight").toarray()
    P = A / A.sum()
    r, c = P.sum(1), P.sum(0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    k = min(2, len(s))
    rows = U[:, :k] * s[:k] / np.sqrt(r)[:, None]
    cols = Vt[:k].T * s[:k] / np.sqrt(c)[:, None]
    pad = np.zeros((1, 2 - k))
    return {n: np.append(xy, pad) for n, xy in zip(sources + targets, np.vstack([rows, cols]))}


def _columns(graph, sweeps=20):
    """Sources in a left column, targets in a right one, each ordered by the weighted mean
    height of its neighbors (barycenter sweeps) to cut edge crossings."""
    sources, targets = _sides(graph)
    strength = dict(graph.degree(weight="weight"))
    y = {n: -i for i, n in enumerate(sorted(sources, key=lambda n: -strength[n]))}

    def place(nodes):
        bary = {n: sum(w * y[m] for m, w in ((m, graph.edges[n, m]["weight"]) for m in graph[n]))
                / strength[n] for n in nodes}
        for i, n in enumerate(sorted(nodes, key=lambda n: -bary[n])):
            y[n] = -i

    for _ in range(sweeps):
        place(targets)
        place(sources)
    pos = {}
    for x, nodes in ((0.0, sources), (1.0, targets)):
        span = max(len(nodes) - 1, 1)
        pos |= {n: np.array([x, 1 + y[n] / span]) for n in nodes}
    return pos


def plot_bipartite_graph(graph, ax=None, source_color="yellowgreen", target_color="#7b3294",
                         node_colors=None, hub_labels=None, legend=None, layout="spring",
                         annotate="nodes", edges=True, seed=0, node_size=12.0, source_size=60.0,
                         spread=1.0, fontsize=8, bold=()):
    """Draw a weighted bipartite graph.

    Layouts:
        ``"spring"``: satellites; the spring layout pulls along each edge in proportion to its
            weight, so targets cluster around their strongest sources.
        ``"biplot"``: correspondence analysis of the weighted biadjacency, both sides placed alike
            in the plane of its first two dimensions; meant for an unpruned graph
            (``threshold=0``), where closeness is the association; pass a pruned graph as
            ``edges`` to draw only its strong links.
        ``"columns"``: sources left, targets right, ordered to cut edge crossings.

    Sources are sized by their weighted degree, edges drawn with width and opacity growing with
    weight.

    Args:
        graph: graph from :func:`bipartite_graph`, :func:`relabel` or :func:`common_subgraph`.
        ax: axes to draw into; a new 7 x 7 in figure when ``None``.
        source_color, target_color: node colors of the two sides.
        node_colors: dict node -> color, overriding its side's color.
        hub_labels: dict node -> text; only these nodes are named.
        legend: dict label -> color for the legend right of the axes; a ``None`` color makes
            the label a bold group header.
        layout: one of ``LAYOUTS``.
        annotate: ``"nodes"`` writes ``hub_labels`` next to their nodes, ``"legend"`` lists them
            in the legend by their node's color instead.
        edges: ``True`` draws the graph's edges, ``False`` none, and a graph over the same
            nodes draws its edges instead, so the layout and the lines can come from different
            prunings.
        seed: spring layout seed.
        node_size: marker area of a target node.
        source_size: marker area of a source with weighted degree 0; it grows with the root.
        spread: scales the spring layout's preferred node distance; above 1 the edges pull less
            against the repulsion, so satellites sit further out.
        fontsize: annotation font size.
        bold: nodes whose names are set in bold.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout!r} not in {LAYOUTS}")
    if annotate not in ANNOTATE:
        raise ValueError(f"annotate {annotate!r} not in {ANNOTATE}")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.set_axis_off()
    if graph.number_of_edges() == 0:
        return ax

    if layout == "spring":
        pos = nx.spring_layout(graph, weight="weight", seed=seed,
                               k=1.5 * spread / np.sqrt(graph.number_of_nodes()))
    else:
        pos = _biplot(graph) if layout == "biplot" else _columns(graph)
    if edges is not False:
        edge_list = list((graph if edges is True else edges).edges(data="weight"))
        w = np.array([e[2] for e in edge_list])
        span = np.ptp(w)
        scaled = (w - w.min()) / span if span else np.ones_like(w)
        colors = [(0.3, 0.3, 0.3, a) for a in 0.15 + 0.45 * scaled]
        nx.draw_networkx_edges(graph, pos, edgelist=[e[:2] for e in edge_list], ax=ax,
                               width=0.3 + 1.2 * scaled, edge_color=colors)

    strength = dict(graph.degree(weight="weight"))
    node_colors = {n: source_color if n[0] == SOURCE else target_color
                   for n in graph} | (node_colors or {})
    for nodes in reversed(_sides(graph)):
        size = [source_size * (1 + np.sqrt(strength[n])) if n[0] == SOURCE else node_size
                for n in nodes]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, ax=ax, node_size=size,
                               node_color=[node_colors[n] for n in nodes],
                               edgecolors="white", linewidths=0.4)

    named = {n: text for n, text in (hub_labels or {}).items() if n in pos}
    entries = dict(legend or {})
    if annotate == "legend":
        entries |= {text: node_colors[n] for n, text in named.items()}
    if entries:
        handles = [Line2D([], [], ls="", marker="o" if color else None, ms=6, mfc=color,
                          mec="white", label=name) for name, color in entries.items()]
        leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1),
                        frameon=False, fontsize=fontsize)
        for text, color in zip(leg.get_texts(), entries.values()):
            if color is None:
                text.set_fontweight("bold")
    if annotate == "nodes":
        for n, text in named.items():
            # Beside the node in columns, outward from the edges; above it otherwise.
            if layout == "columns":
                left = n[0] == SOURCE
                place = {"ha": "right" if left else "left", "va": "center",
                         "xytext": (-8 if left else 8, 0)}
            else:
                place = {"ha": "center", "va": "bottom", "xytext": (0, 6)}
            ax.annotate(text, pos[n], fontsize=fontsize, textcoords="offset points",
                        fontweight="bold" if n in bold else "normal",
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none",
                              "alpha": 0.7}, **place)
    return ax
