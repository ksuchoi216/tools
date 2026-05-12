from IPython.display import Image, display


def show_graph(graph, is_save=False, save_path="graph.png"):
    img = graph.get_graph().draw_mermaid_png()
    display(Image(img))
    if is_save:
        with open(save_path, "wb") as f:
            f.write(img)
