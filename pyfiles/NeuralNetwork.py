import numpy as np


def Relu(x):
    return max(0, x)


def htan(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def no_af(x):
    return x


class _node(object):
    def __init__(self, id: int, bias: float, layer:int):
        self.id = id
        self.bias = bias
        self.layer: int = layer

        self.value: float = 0
        match layer:
            case 0:
                self.activation = no_af
            case 1:
                self.activation = Relu
            case 2:
                self.activation = htan
            case _:
                raise ValueError(f'{layer} is invalid layer number')

    def output(self):
        return self.activation(self.value + self.bias)


'''
GENOTYPE DICTIONARY FORMAT
Nodes = { key=id: int || value={ bias: float, layer: int = 0|1|2 } }
Connections = { key=edge: (in: int, out: int) || value={ weight: float, enabled: bool, innovation: int } }
'''


class NeuralNetwork(object):
    def __init__(self, genotype: dict = None):
        self.nodes = {}
        self.dependencies = {}

        self.outputs = []
        self.topology = []
        self.activations = {}

        if not genotype:
            raise ValueError('Genotype not given')
        else:
            self.genotype = genotype

        self.create()
        self.topological_sort()

    def create(self):
        nodes: dict = self.genotype['nodes']
        connections: dict = self.genotype['connections']

        for id, values in nodes.items():
            # create node and dependency list for all nodes
            if values['layer'] == 0:
                bias = 0
            else:
                bias = values['bias']
            self.nodes[id] = _node(id, bias, values['layer'])
            if values['layer'] == 2:
                self.outputs.append(id)

        for edge, values in connections.items():
            # if not enabled skip connection
            if values['enabled']:
                inp, outp = edge
                weight = values['weight']

                if outp not in self.dependencies:
                    self.dependencies[outp] = []
                self.dependencies[outp].append( (inp, weight) )

    def reset(self):
        for _, node in self.nodes.items():
            node.value = 0

    def forward(self, inputs):
        for i in range(len(inputs)):
            self.nodes[i].value = inputs[i]
            self.activations[i] = inputs[i]

        for idx in self.topology:
            deps = self.dependencies[idx]
            if not deps:
                continue

            for d_id, d_weight in deps:
                activation = self.nodes[d_id].output() * d_weight
                self.nodes[idx].value += activation
                self.activations[idx] += activation

        output = []
        for idx in self.outputs:
            activation = self.nodes[idx].output()
            output.append(activation)
            self.activations[idx] = activation

        self.reset()
        return output

    def topological_sort(self):
        dependency_set = set(self.dependencies.keys())
        topology = []

        while dependency_set:
            for i in dependency_set.copy():
                # get dependancy list
                depends = self.dependencies[i]
                if not depends:
                    # if no dependencies add to topology
                    dependency_set.remove(i)
                    # topology.append(i)

                else:
                    no_deps = True
                    # iterate over dependancy list
                    for id, _ in depends:
                        if id in dependency_set:
                            # if dependacy still exists skip
                            no_deps = False
                            break
                    if no_deps:
                        # if no dependencies add to topology
                        dependency_set.remove(i)
                        topology.append(i)
        self.topology = topology

    def __call__(self, inpt):
        return self.forward(inpt)

    def layers(self):
        depth = {}
        # process input layer
        for ix, node in self.nodes.items():
            if node.layer == 0:
                depth[ix] = 0

        for ix in self.topology:
            # only process hidden layer
            if self.nodes[ix].layer == 2:
                continue

            depends = self.dependencies[ix]
            if not depends:
                continue

            depends_depths = [0]
            for id, _ in depends:
                if id not in self.topology:
                    continue
                depends_depths.append(depth[id])

            depth[ix] = max(depends_depths) + 1

        max_depth = max(depth.values())
        # process output layer
        for ix, node in self.nodes.items():
            if node.layer == 2:
                depth[ix] = max_depth + 1
        return depth


if __name__ == '__main__':
    pass
