from pyfiles.NeuralNetwork import NeuralNetwork


def get_nn_pos(genotype, width, height) -> dict[int:tuple[int]]:
    nn = NeuralNetwork(genotype=genotype)
    node_layer = nn.layers()
    num_layers = max(node_layer.values()) + 1
    layer_nodes: dict[int:list] = {}
    for i in range(num_layers):
        layer_nodes[i] = []
        for id, layer in node_layer.items():
            if layer == i:
                layer_nodes[i].append(id)

    x_spacing = width / (num_layers - 1)

    node_pos: dict[int:tuple[int]] = {}
    for layer, nodes in layer_nodes.items():
        xpos = x_spacing*layer
        yspacing = height / (len(nodes) + 1)
        for i in range(len(nodes)):
            ypos = yspacing*(i+1)
            node_pos[nodes[i]] = (xpos, ypos)

    return node_pos