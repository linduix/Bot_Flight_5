import sys
from pyfiles.droneV1 import DroneData
import numpy as np
import random

# GLOBALS SETUP
WIDTH, HEIGHT = 1600, 900
WORLDSCALE = 20
startpos = [WIDTH/(2*20), HEIGHT/(2*20)]


def dependencies(connections: dict, node: int, memo=None) -> set[int]:
    if memo is None:
        memo: dict[int:set] = dict()

    deps = set()
    for inp, outp in connections.keys():
        if outp == node:
            # check memo for dependencies
            if inp in memo:
                deps.update(memo[inp])
                deps.add(inp)
            else:
                deps.add(inp)
                deps.update(dependencies(connections, inp))

    memo[node] = deps
    return deps


'''
GENOTYPE DICTIONARY FORMAT
Nodes = { key=id: int || value={ bias: float, layer: int = 0|1|2 } }
Connections = { key=edge: (in: int, out: int) || value={ weight: float, enabled: bool, innovation: int } }
'''


def distance(g1: dict, g2: dict):
    c1 = 1
    c2 = 1
    c3 = 0.4

    g1_innovation = set([v['innovation'] for v in g1['connections'].values()])
    g2_innovation = set([v['innovation'] for v in g2['connections'].values()])
    g1_innovation.add(0)
    g2_innovation.add(0)

    excess_threshold = min(max(g1_innovation), max(g2_innovation))
    # Number of Genes
    N = max(len(g1_innovation), len(g2_innovation))

    # Disjoints and Excess
    D = 0
    E = 0
    uncommon = g1_innovation ^ g2_innovation
    for gene in uncommon:
        if gene > excess_threshold:
            E += 1
        else:
            D += 1

    # Weighted Difference
    W = 0
    common = set(g1['connections'].keys()) & set(g2['connections'].keys())
    for gene in common:
        W += np.abs(g1['connections'][gene]['weight'] - g2['connections'][gene]['weight'])

    return c1*E/N + c2*D/N + c3*W


'''
GENOTYPE DICTIONARY FORMAT
Nodes = { key=id: int || value={ bias: float, layer: int = 0|1|2 } }
Connections = { key=edge: (in: int, out: int) || value={ weight: float, enabled: bool, innovation: int } }
'''


def crossover(p1: DroneData, p2: DroneData) -> dict:
    # score select primary parent
    best_parent: DroneData = max(p1, p2, key=lambda x: x.score)
    worse_parent: DroneData = min(p1, p2, key=lambda x: x.score)

    common = set(p1.genotype['connections'].keys()).intersection(set(p2.genotype['connections'].keys()))

    selected_connections = {}
    for edge in best_parent.genotype['connections'].keys():
        # randomly select for common edges
        if edge in common:
            if np.random.random() < 0.5:
                selected_connections[edge] = best_parent.genotype['connections'][edge]
            else:
                selected_connections[edge] = worse_parent.genotype['connections'][edge]

        # select from better parent for disjoint and excess genes
        else:
            selected_connections[edge] = best_parent.genotype['connections'][edge]

        # 25% chance to reenable
        if selected_connections[edge]['enabled']:
            if np.random.random() < 0.25:
                selected_connections[edge]['enabled'] = True

    node_set = set()
    for node, value in p1.genotype['nodes'].items():
        # add node if its input or output
        if value['layer'] != 1:
            node_set.add(node)
    for inp, outp in selected_connections.keys():
        node_set.add(inp)
        node_set.add(outp)

    # Chose node values from parents randomly
    selected_nodes = {}
    for id in node_set:
        if np.random.random() < 0.5:
            option1 = p1
            option2 = p2
        else:
            option1 = p2
            option2 = p1

        if id in option1.genotype['nodes'].keys():
            selected_nodes[id] = option1.genotype['nodes'][id]
        else:
            selected_nodes[id] = option2.genotype['nodes'][id]

    return {'nodes': selected_nodes, 'connections': selected_connections}


'''
GENOTYPE DICTIONARY FORMAT
Nodes = { key=id: int || value={ bias: float, layer: int = 0|1|2 } }
Connections = { key=edge: (in: int, out: int) || value={ weight: float, enabled: bool, innovation: int } }
'''


def mutate(genotype: dict, innovations: dict) -> (dict, dict):
    # mutation values
    value_shift = 0.8
    delete_chance = 0.01
    new_node = 0.03
    new_connection = 0.05

    # value change
    if np.random.random() < value_shift:
        # 50/50 between bias or weight change
        if np.random.random() < 0.5:
            # 90% chance for each weight to shift
            for value in genotype['connections'].values():
                if np.random.random() < 0.9:
                    value['weight'] += np.random.normal(scale=0.1)
                # # 1% chance for complete random value
                # else:
                #     value['weight'] = np.random.normal(loc=0, scale=0.3)
        else:
            # 90% chance for bias shift
            for value in genotype['nodes'].values():
                if np.random.random() < 0.9:
                    value['bias'] += np.random.normal(scale=1)
                # # 1% chance for random value
                # else:
                #     value['bias'] = np.random.normal()

    # delete chance
    if np.random.random() < delete_chance:
        for edge in list(genotype['connections'].keys()):
            # 10% chance for each connection
            if np.random.random() > 0.1:
                continue
            del genotype['connections'][edge]
            break

    if np.random.random() < 0.00:
        for edge in list(genotype['connections'].keys()):
            # 10% chance for each connection
            if np.random.random() > 0.1:
                continue
            genotype['connections'][edge]['enabled'] = False
            break

    # delete node
    hidden_nodes = [n for n, val in genotype['nodes'].items() if val['layer'] == 1]
    if len(hidden_nodes) > 0:
        if random.random() < 0.01:
            n = random.choice(hidden_nodes)
            del genotype['nodes'][n]
            for edge in list(genotype['connections'].keys()):
                if n == edge[0] or n == edge[1]:
                    del genotype['connections'][edge]

    # new nodes
    if len(genotype['connections'].keys()) > 1:
        # if at least 1 connection
        if np.random.random() < new_node:
            edge = random.choice(list(genotype['connections'].keys()))

            max_node = max(genotype['nodes'].keys())
            old_weight = genotype['connections'][edge]['weight']

            # create new node
            n_id = max_node + 1
            n = {'bias': 0, 'layer': 1}

            # create new connections
            edge1 = (edge[0], n_id)
            if edge1 in innovations.keys():
                i1 = innovations[edge1]
            else:
                i1 = innovations['max'] + 1
                innovations['max'] += 1
                innovations[edge1] = i1
            c1 = {'weight': 1, 'enabled': True, 'innovation': i1}

            edge2 = (n_id, edge[1])
            if edge2 in innovations.keys():
                i2 = innovations[edge2]
            else:
                i2 = innovations['max'] + 1
                innovations['max'] += 1
                innovations[edge2] = i2
            c2 = {'weight': old_weight, 'enabled': True, 'innovation': i2}

            # disable old connections
            del genotype['connections'][edge]

            # add new connections and nodes
            genotype['nodes'][n_id] = n
            genotype['connections'][edge1] = c1
            genotype['connections'][edge2] = c2

    # new connection chance 5%
    if np.random.random() < new_connection:
        # input/output selection
        inp_choices = []
        outp_choices = []
        for id, value in genotype['nodes'].items():
            if value['layer'] != 2:
                inp_choices.append(id)
            if value['layer'] != 0:
                outp_choices.append(id)
        inp = np.random.choice(inp_choices)

        # delete self from output choices
        if inp in outp_choices:
            outp_choices.remove(inp)

        # make sure to avoide cicular dependencies
        inp_deps = dependencies(genotype['connections'], inp)
        for dep in inp_deps:
            if dep in outp_choices:
                outp_choices.remove(dep)
        outp = np.random.choice(outp_choices)

        # check edge innovation
        edge = (inp, outp)
        if edge in innovations.keys():
            innovation = innovations[edge]
        else:
            innovation = innovations['max'] + 1
            innovations['max'] += 1
            innovations[edge] = innovation

        # add new connection
        genotype['connections'][edge] = {'weight': np.random.random(), 'enabled': True, 'innovation': innovation}

    return genotype, innovations


def next_generation(drones: list[DroneData], gen_size, innovations: dict, species: dict, thresh: int):
    all_drones: list[DroneData] = sorted(drones, key=lambda x: x.score, reverse=True)  # sort from high to low score
    # all_drones: list[DroneData] = drones
    species_drones: list[list[DroneData]] = [[] for _ in range(len(species))]

    stagnant = list(species.keys())
    # speciate
    for drone in all_drones:
        match = False
        for s, values in species.items():
            # get representative
            rep = values['rep']
            if distance(drone.genotype, rep) < thresh:
                # if < threshold: flag match, add drone to species list, and set drone species
                match = True
                species_drones[s].append(drone)
                drone.species = s
                # check if fitnes score has increased and remove stagnant flag
                if drone.score > values['max_fit'] and s in stagnant:
                    stagnant.remove(s)
                    values['stag_ticker'] = 0
                break

        # create species if not match others
        if not match:
            k = len(species)
            species[k] = {'rep': drone.genotype, 'max_fit': 0, 'stag_ticker': 0, 'stagnant': False}
            species_drones.append([drone])
            drone.species = k
    # increcment stagnant species
    for s in stagnant:
        species[s]['stag_ticker'] += 1
        if species[s]['stag_ticker'] >= 10:
            species[s]['stagnant'] = True

    # fitness sharing
    for drone in all_drones:
        drone.score /= len(species_drones[drone.species])

        # if species stagnated set fitness to 0
        if species[drone.species]['stagnant']:
            drone.score = 0

    elite: list[DroneData] = []
    for ds in species_drones:
        # 30% best of each species are elite
        ratio = int(np.ceil(len(ds) * 0.2))
        elite.extend(ds[:ratio])

    if len(elite) > gen_size:
        elite = elite[:gen_size]

    elite_species = [[] for _ in range(len(species))]
    for drone in elite:
        elite_species[drone.species].append(drone)

    # selection weights for the drones
    elite_weights: list[float] = []
    for x in elite:
        # layers = max(x.brain.layers().values())
        # adjusted_score = x.score**1.1 - 10*np.exp(0.005*layers) + 11
        # adjusted_score = (x.score**1.1)*np.exp(-np.exp(0.6*layers)/100)
        # elite_weights.append(adjusted_score)
        elite_weights.append(x.score)
    all_weights: list[float] = [x.score for x in all_drones]
    species_weights: list[list[float]] = [[x.score for x in ds] for ds in species_drones]

    if np.sum(elite_weights) == 0:
        elite_weights = [1 for _ in range(len(elite_weights))]
    if np.sum(all_weights) == 0:
        all_weights = [1 for _ in range(len(all_weights))]

    # MAIN
    next_gen_genes: list[dict] = []
    next_gen_genes.extend([d.genotype for d in elite])
    while len(next_gen_genes) < gen_size:
        # SELECTION
        try:
            p1: DroneData = random.choices(elite, weights=elite_weights)[0]
        except ValueError:
            print('Zero Sum Weights', elite_weights)
            sys.exit()
        # 1% chance for interspecies reproduction
        if np.random.random() < 0.04:
            p2: DroneData = random.choices(all_drones, weights=all_weights)[0]
        else:
            try:
                p2: DroneData = random.choices(species_drones[p1.species], weights=species_weights[p1.species])[0]
            except ValueError:
                weights = [1 for _ in range(len(species_weights[p1.species]))]
                p2: DroneData = random.choices(species_drones[p1.species], weights=weights)[0]

        # CROSSOVER
        child_genotype: dict = crossover(p1, p2)

        # Mutation
        child_genotype, innovations = mutate(child_genotype, innovations)

        # Adding child
        next_gen_genes.append(child_genotype)

    return next_gen_genes, innovations, species


def get_base():
    base_genotype = {'nodes': {}, 'connections': {}}
    inputs, outputs = 8, 2
    for i in range(inputs + outputs):
        if i < inputs:
            base_genotype['nodes'][i] = {'bias': 0, 'layer': 0}
        else:
            base_genotype['nodes'][i] = {'bias': 0, 'layer': 2}
    return base_genotype
