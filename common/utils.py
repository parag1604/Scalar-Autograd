"""
Helpers shared by all versions: drawing the compute graph with graphviz.

Rendering needs BOTH the `graphviz` python package (pip) and the Graphviz
system binary `dot` (e.g. `brew install graphviz`, `sudo apt install graphviz`
or https://graphviz.org/download/). If either is missing we print a hint and
carry on, so the rest of the tutorial still runs.
"""
import os


def trace(root):
    """Collect every node and edge reachable from `root` (depth-first)."""
    nodes, edges = set(), set()
    def build(node):
        if node not in nodes:
            nodes.add(node)
            for child in node._prev:
                edges.add((child, node))
                build(child)
    build(root)
    return nodes, edges


def draw_compute_graph(root, format='svg', rankdir='LR', requires_grad=True):
    from graphviz import Digraph  # imported lazily: optional dependency

    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    G = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        # one box per Variable: its label (if any), value and gradient
        fields = [n.label] if n.label else []
        fields.append("data: %.4f" % n.data)
        if requires_grad:
            fields.append("grad: %.4f" % n.grad)
        G.node(name=str(id(n)),
               label="{ " + " | ".join(fields) + " }", shape='record')
        # one small node per operation, pointing at the Variable it produced
        if n._op:
            G.node(name=str(id(n)) + n._op, label=n._op)
            G.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        G.edge(str(id(n1)), str(id(n2)) + n2._op)

    return G


def render_graph(root, name, requires_grad=True, out_dir='graphs'):
    """Draw the graph of `root` to <out_dir>/<name>.svg if graphviz is set up."""
    try:
        import graphviz
        G = draw_compute_graph(root, requires_grad=requires_grad)
        path = G.render(os.path.join(out_dir, name), cleanup=True)
        print(f"Compute graph written to {path}")
    except ImportError:
        print("Skipping the graph: `pip install graphviz` to draw it.")
    except graphviz.ExecutableNotFound:
        print("Skipping the graph: the Graphviz `dot` binary is not installed "
              "(see https://graphviz.org/download/).")
