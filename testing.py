from pyfiles.droneV1 import AiDrone
import numpy as np
import pygame as pg
import json
import sys
from pyfiles.network import get_nn_pos
import ast
import colorsys

# Initialize Pygame
pg.init()

# Set up the screen
WIDTH, HEIGHT = 1600, 900
screen = pg.display.set_mode((WIDTH, HEIGHT), pg.SRCALPHA)
pg.display.set_caption("Pygame Boilerplate")
WORLDSCALE = 20

amt = 10
ids = []
drones = []
# drone and target setup
with open('data/species.json', 'r') as f:
    data: dict = json.load(f)
    for s, val in data.items():
        gene = val['rep']
        gene['connections'] = {ast.literal_eval(k): v for k, v in gene['connections'].items()}
        gene['nodes'] = {int(k): v for k, v in gene['nodes'].items()}
        ids.append(int(s))
        drones.append(AiDrone(genotype=gene, startpos=[WIDTH/(WORLDSCALE * 2), HEIGHT/(WORLDSCALE * 2)]))
ids = ids[:amt]
drones = drones[:amt]
idmax = max(ids)

font = pg.font.SysFont('freesansbold', 24)
track_font = pg.font.SysFont('freesansbold', 48)

drone_sprite = pg.Surface([20, 20])
drone_sprite.fill('black')
pg.draw.rect(drone_sprite, 'white', [1, 1, 18, 18])
pg.draw.circle(drone_sprite, 'red', (10, 2), 2)


def l2_sim():
    dt = 16 / 1000
    target = np.random.rand(1, 2)[0] * np.array([WIDTH // WORLDSCALE, HEIGHT // WORLDSCALE])
    w, h = 400, 300
    nn_surf = pg.Surface((w, h))
    nn_surf.set_colorkey('black')
    best_score = -1

    counter = 0
    while True:
        # Event handling
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    return

        # Clear the screen
        screen.fill('black')

        # Draw nn
        if counter == 0:
            # Skip first frame
            counter += 1
        else:
            max_drone = max(drones, key=lambda x: x.completed)
            first_place = max_drone if max_drone.completed > best_score else first_place
            best_score = first_place.completed
            draw_nn(first_place, w, h, nn_surf)
            screen.blit(nn_surf, (0, 0))

            id = ids[drones.index(first_place)]
            rgb = colorsys.hsv_to_rgb(id*0.8/idmax, 0.9, 1)
            color = [int(v*255) for v in rgb]
            track_text = track_font.render(f'Tracking {id}: {first_place.completed}',
                                           True, color)
            screen.blit(track_text, (10, h+10))

        if all([d.crash for d in drones]):
            return

        # Update Sim
        for id, drone in zip(ids, drones):
            rgb = colorsys.hsv_to_rgb(id*0.8/idmax, 0.9, 1)
            color = [int(v*255) for v in rgb]

            drone.process(target)
            drone.update(dt)

            new_targ = False
            dist = np.linalg.norm(drone.pos - target)
            if dist < 0.5:
                drone.touch_time += dt
                if drone.touch_time > 1:
                    drone.completed += 1
                    new_targ = True
            elif dist > 100:
                drone.crash = True
                drone.done = True
            else:
                drone.touch_time = 0

            if new_targ:
                while np.linalg.norm(drone.pos - target) < 40*20/WORLDSCALE or np.linalg.norm(drone.pos - target) > 60*20/WORLDSCALE:
                    ratio = np.array([np.random.random(), np.random.random()])
                    target = ratio * np.array([WIDTH // WORLDSCALE, HEIGHT // WORLDSCALE])

            # Calculations
            center: np.ndarray = drone.pos.copy() * WORLDSCALE
            center[1] = HEIGHT - center[1]
            center = center.astype(np.int64)

            # Draw here
            drone_sprite.fill('black')
            pg.draw.rect(drone_sprite, color, [1, 1, 18, 18])
            pg.draw.circle(drone_sprite, 'red', (10, 2), 2)
            rotated_sprite = pg.transform.rotate(drone_sprite, np.degrees(drone.angle))
            rotated_sprite.set_colorkey('black')
            sprite_rect = rotated_sprite.get_rect(center=center)
            screen.blit(rotated_sprite, sprite_rect)

            pos = (int(center[0]), int(center[1]-(10+18)))
            id_text = font.render(f'{id}', True, color)
            text_rect = id_text.get_rect(center=pos)
            screen.blit(id_text, text_rect)

        target_pos = target * WORLDSCALE
        target_pos[1] = HEIGHT - target_pos[1]
        pg.draw.circle(screen, 'red', target_pos, 3)

        # Update the display
        pg.display.flip()

        # Cap the frame rate
        dt = pg.time.Clock().tick(60) / 1000


def manual_sim():
    t_time = 0
    dt = 16 / 1000
    pg.mouse.set_visible(False)
    w, h = 400, 200
    nn_surf = pg.Surface((w, h))
    nn_surf.set_colorkey('black')

    while True:
        # Event handling
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    return

        # Clear the screen
        screen.fill('black')

        if all([d.crash for d in drones]):
            return

        # Get mousepos
        mouse_pos = np.array(pg.mouse.get_pos())
        mouse_pos[1] = HEIGHT - mouse_pos[1]
        target = mouse_pos/WORLDSCALE

        # Update Sim
        for id, drone in zip(ids, drones):
            rgb = colorsys.hsv_to_rgb(id*0.8/idmax, 0.9, 1)
            color = [int(v*255) for v in rgb]

            drone.process(target)
            drone.update(dt)

            if np.linalg.norm(drone.pos - target) > 100:
                drone.crash = True

            # Calculations
            center = drone.pos.copy() * WORLDSCALE
            center[1] = HEIGHT - center[1]

            # Draw here
            drone_sprite.fill('black')
            pg.draw.rect(drone_sprite, color, [1, 1, 18, 18])
            pg.draw.circle(drone_sprite, 'red', (10, 2), 2)
            rotated_sprite = pg.transform.rotate(drone_sprite, np.degrees(drone.angle))
            rotated_sprite.set_colorkey('black')
            sprite_rect = rotated_sprite.get_rect(center=center)
            screen.blit(rotated_sprite, sprite_rect)

            pos = (int(center[0]), int(center[1]-(10+18)))
            id_text = font.render(f'{id}', True, color)
            text_rect = id_text.get_rect(center=pos)
            screen.blit(id_text, text_rect)

        target_pos = target * WORLDSCALE
        target_pos[1] = HEIGHT - target_pos[1]
        pg.draw.circle(screen, 'red', target_pos, 3)

        draw_nn(max(drones, key=lambda x: x.completed), w, h, nn_surf)
        screen.blit(nn_surf, (0, 0))

        # Update the display
        pg.display.flip()

        # Cap the frame rate
        dt = pg.time.Clock().tick(60) / 1000
        t_time += dt


def draw_nn(drone: AiDrone, w, h, surf: pg.Surface):
    gene = drone.genotype
    displayw = w
    displayh = h

    paddingx = 20
    paddingy = 0
    node_pos = get_nn_pos(gene, displayw - paddingx * 2, displayh - paddingy * 2)
    intesity = 10

    surf.fill('black')
    try:
        max_weight = np.abs(np.array([v['weight'] for v in gene['connections'].values()])).max()
    except ValueError:
        max_weight = 0
    for (inp, outp), vals in gene['connections'].items():
        if inp not in node_pos or outp not in node_pos:
            continue
        weight = vals['weight']
        activation = np.clip(drone.brain.activations[inp], -1, 1)

        if activation >= 0:
            color = (intesity, (255 - intesity) * activation + intesity, intesity)
        else:
            color = ((255 - intesity) * -activation + intesity, intesity, intesity)

        if vals['enabled'] is False:
            color = 'black'

        inx, iny = node_pos[inp]
        outx, outy = node_pos[outp]

        weight = int(np.ceil(np.abs(weight / max_weight) * 6))
        pg.draw.line(surf, color, (inx + paddingx, iny + paddingy),
                     (outx + paddingx, outy + paddingy), weight)

    for n, pos in node_pos.items():
        screen_pos = (pos[0] + paddingx, pos[1] + paddingy)
        activation = np.clip(drone.brain.activations[n], -1, 1)
        if activation >= 0:
            color = (intesity, (255 - intesity) * activation + intesity, intesity)
        else:
            color = ((255 - intesity) * -activation + intesity, intesity, intesity)
        pg.draw.circle(surf, color, screen_pos, 10)

    ############
    '''
    shape = drone.brain.shape
    width = 250
    gap = 100

    for i, layer_amt in enumerate(shape):
        xpos = gap * (i+1) - (gap-30)
        for ix in range(layer_amt):
            ypos = HEIGHT + (width / (layer_amt + 1)) * (ix + 1) - width

            # Draw weight lines:
            intesity = 10
            if i < len(shape) - 1:
                xpos2 = gap * (i + 2) - (gap-30)
                for jx in range(shape[i + 1]):
                    layer2_amt = shape[i + 1]
                    ypos2 = HEIGHT + (width / (layer2_amt + 1)) * (jx + 1) - width

                    activation = drone.brain.weight_activations[i][jx, ix]
                    weights = drone.brain.layers[i].weights
                    line_size = np.abs(weights[jx, ix]) * 4 / np.abs(weights[jx, ix]).max()
                    if activation >= 0:
                        color = (intesity, (255-intesity)*activation+intesity, intesity)
                    else:
                        color = ((255-intesity)*-activation+intesity, intesity, intesity)

                    pg.draw.line(screen, color, [xpos, ypos], [xpos2, ypos2], width=int(line_size))

            # Draw Node activations
            activation = drone.brain.node_activations[i][ix, 0]
            if activation >= 0:
                color = (intesity, (255 - intesity) * activation + intesity, intesity)
            else:
                color = ((255 - intesity) * -activation + intesity, intesity, intesity)

            pg.draw.circle(screen, color, (xpos, ypos), 8)
    '''


# manual_sim()
l2_sim()

# Quit Pygame
pg.quit()
sys.exit()
