import sys
import ast
import pyfiles.geneticFuncs
from pyfiles.droneV1 import DroneData
from pyfiles.geneticFuncs import next_generation, get_base
from pyfiles.network import get_nn_pos
import signal
import time
import json
import numpy as np
import os
import shutil
import subprocess
import pygame as pg


def initializer():
    """Ignore SIGINT in child workers."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def save(gen, best_gene: dict, drone_genes: dict, innovations, species, dist_threshold):
    # Make backup
    if os.path.exists('data/data.json'):
        shutil.copy('data/data.json', 'backup/data.json')

    # try:
    # Save the best drone
    path = f'data/data.json'
    with open(path, 'w') as f:
        # Reformat Best Drone
        best_genotype: dict = best_gene
        best_genotype['connections'] = {str(k): v for k, v in best_genotype['connections'].items()}
        best_genotype['nodes'] = {int(k): v for k, v in best_genotype['nodes'].items()}

        # Reformat All Drones
        for drone in drone_genes:
            drone['connections'] = {str(k): v for k, v in drone['connections'].items()}
            drone['nodes'] = {int(k): v for k, v in drone['nodes'].items()}

        # Reformat Innovations
        innovations = {str(k): v for k, v in innovations.items()}

        # Save Data
        data = {'generation': gen, 'best': best_genotype, 'all': drone_genes,
                'innovations': innovations, 'thresh': dist_threshold}
        json.dump(data, f)

    # output species for testing purposes
    with open('data/species.json', 'w') as f:
        for s, data in species.items():
            rep = data['rep']
            rep['nodes'] = {int(k): v for k, v in rep['nodes'].items()}
            rep['connections'] = {str(k): v for k, v in rep['connections'].items()}
        json.dump(species, f)


def load_results():
    pass


# GLOBALS SETUP
WIDTH, HEIGHT = 1600, 900
displayw = 800
displayh = 600

paddingx = 50
paddingy = 10

starpos = [1600/40, 1600/40]

# Training settings
gen_size: int = 5000   # Drones per generation
gen_threshold: int = 5_000  # Total generations to train


if __name__ == '__main__':
    # Setup PG
    pg.init()
    SCREEN = pg.display.set_mode((displayw, displayh))

    # Load Data
    targs: int = 15
    try:
        with open('data/data.json', 'r') as f:
            data = json.load(f)

            best_gene = data['best']

            drone_genes = data['all']

            # Format innovations
            innovations = {}
            for k, v in data['innovations'].items():
                if k != 'max':
                    innovations[ast.literal_eval(k)] = v
                else:
                    innovations[k] = v

            gen = data['generation']
            dist_threshold = data['thresh']

    except json.decoder.JSONDecodeError as e:
        os.remove('data/data.json')

        if os.path.exists('backup/data.json'):
            with open('backup/data.json', 'r') as f:
                data = json.load(f)

                best_gene = data['best']

                drone_genes = data['all']

                # Format innovations
                innovations = {}
                for k, v in data['innovations'].items():
                    if k != 'max':
                        innovations[ast.literal_eval(k)] = v
                    else:
                        innovations[k] = v

                gen = data['generation']
                dist_threshold = data['thresh']
        else:
            print(f"Error: {e}")
            sys.exit()

    except FileNotFoundError:
        drone_genes = [get_base() for _ in range(gen_size)]
        best_gene = drone_genes[0]
        innovations = {'max': 0}
        dist_threshold = 3
        gen = 0

    # training settings
    starting_gen = gen
    species = {}
    target = int(np.sqrt(gen_size))
    reset_timer = 0
    t_thresh = 20

    # Check if already Done
    if gen == gen_threshold:
        print('\033[32mTraining Done\033[0m')
        sys.exit()

    # Training loop
    try:
        while gen < gen_threshold:
            gen += 1
            reset_timer += 1

            t_thresh *= 1.0019
            t_thresh = min(t_thresh, 20)

            # Event handling
            for event in pg.event.get():
                match event.type:
                    case pg.QUIT:
                        pg.quit()
                        raise KeyboardInterrupt

            # output current genes to json file
            with open('data/drones.json', 'w') as f:
                # format genes
                for drone in drone_genes:
                    drone['connections'] = {str(k): v for k, v in drone['connections'].items()}
                    drone['nodes'] = {int(k): v for k, v in drone['nodes'].items()}
                json.dump(drone_genes, f)

            start_time = time.time()
            # Run rust scorer
            result = subprocess.run(['./target/release/bot_flight_5.exe'], capture_output=True, text=True)
            if result.returncode != 0:
                print("Execution Failed:", result.returncode)
                print(result.stderr)
                sys.exit()
            end_time = time.time()

            # Load results
            drones: list[DroneData] = []
            with open('data/results.json', 'r') as f:
                data = json.load(f)

                for entry in data:
                    if type(entry['score']) is not float:
                        score = 0
                    else:
                        score = entry['score']
                    d = DroneData(
                        genotype=entry['drone'],
                        score=float(score),
                        crash=bool(entry['crash']),
                        completed=int(entry['completed']),
                        survived=float(entry['survived'])
                    )
                    drones.append(d)

            best_drone: DroneData = max(drones, key=lambda x: x.score)
            node_pos = get_nn_pos(best_drone.genotype, displayw - paddingx * 2, displayh - paddingy * 2)

            # Update Results
            # if (gen == starting_gen + 1) or (end_time - start_time >= 5):
            print(f"Gen:  {gen:>4} | "
                  # f"Score: {np.mean([x.score for x in drones]):0>6.2f} | "
                  f"Score: {best_drone.score:0>6.2f} | "
                  f"{'[CRASH]' if best_drone.crash else '[ALIVE]':<7} {best_drone.survived:0>5.2f}s | "
                  f"Targets [{best_drone.completed:0>2}/{targs:0>2}] | "
                  f"{end_time - start_time:0>5.2f}s | Threshold: {dist_threshold:.1f} | "
                  f"Species: {len(species.keys())}")

            # Draw Network Visualization
            SCREEN.fill('black')
            try:
                max_weight = np.abs(np.array([v['weight'] for v in best_drone.genotype['connections'].values()])).max()
            except ValueError:
                max_weight = 0
            for (inp, outp), vals in best_drone.genotype['connections'].items():
                if inp not in node_pos or outp not in node_pos:
                    continue
                if vals['weight'] > 0:
                    color = 'green'
                else:
                    color = 'red'
                if vals['enabled'] is False:
                    color = 'white'

                inx, iny = node_pos[inp]
                outx, outy = node_pos[outp]

                weight = int(np.ceil(np.abs(vals['weight'] / max_weight) * 5)) + 1
                pg.draw.line(SCREEN, color, (inx + paddingx, iny + paddingy),
                             (outx + paddingx, outy + paddingy), weight)

            for n, pos in node_pos.items():
                screen_pos = (pos[0] + paddingx, pos[1] + paddingy)
                pg.draw.circle(SCREEN, 'white', screen_pos, 10)

            # Update display:
            pg.display.flip()

            # Get the next generation
            drone_genes, innovations, species = next_generation(drones, gen_size, innovations, species,
                                                               thresh=dist_threshold)

            # Reset species
            if reset_timer > 25 or target/2 > len(species) or len(species) > target*2:
                # Adjust threshold
                if len(species) > target:  # and dist_threshold < 4:
                    dist_threshold += min(0.5 * len(species) / target, 0.25*dist_threshold)
                if len(species) < target and dist_threshold > 0.5:
                    dist_threshold -= min(0.1 * target / max(len(species), 1), 0.25*dist_threshold)
                dist_threshold = max(dist_threshold, 0.5)
                species = {}
                reset_timer = 0

        print('\033[33mTraining Done\033[0m')

    # Check for Keyboard Interrupt
    except KeyboardInterrupt:
        print('\033[33mInterrupt Recieved\033[0m')

    # Save Data
    finally:
        print('\033[33mSaving Drones...\033[0m')
        save(gen, best_gene, drone_genes, innovations, species, dist_threshold)
        print('\033[32mDone\033[0m')

