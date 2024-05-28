use crate::drone::Drone;
use rand::Rng;
use rand::rngs::ThreadRng;

fn euclid_distance(vector: &[f64; 2]) -> f64 {
    return (vector[0].powi(2) + vector[1].powi(2)).sqrt()
}

fn score(drone: &mut Drone, target: &Target, dt: f64) {
    let max_distance = 100.;
    let distance = euclid_distance(&[target.pos[0] - drone.pos[0],
                                        target.pos[1]-drone.pos[1]]);

    // get points based on timed distance upto 1
    if drone.score < 1. {
        drone.score += dt/distance
    }

    // if drone in touch range
    if distance < 0.5 {
        drone.touch_time += dt;
        if drone.touch_time > 1. {
            // if touched for 1 second
            drone.score += target.points / drone.completion_time;
            drone.completed += 1;
            drone.completion_time = 0.
        }
    // else if within max distance
    } else if distance < max_distance {
        drone.completion_time += dt;
    } else {
        drone.crash = true;
        drone.done = true;
    }

}

struct Target {
    points: f64,
    pos: [f64; 2]
}
impl Target {
    fn new(old_pos: [f64; 2], rng: &mut ThreadRng) -> Target {
        // arbitrary training world size decided from pygame window size
        let length = 1600./20.;

        let ratiox: f64 = rng.gen();
        let ratioy: f64 = rng.gen();

        let x = length * ratiox;
        let y = length * ratioy;
        let pos = [x, y];

        let points = euclid_distance(&[x - old_pos[0], y-old_pos[1]]);
        Target {points, pos}
    }
}

pub fn run_level(drones: &mut [Drone]) {
    // level settings
    let dt = 16./1000.;
    let mut t_time = 0.;
    let t_thresh = 60.;
    let target_amt = 15u32;

    // random generator
    let mut rng = rand::thread_rng();

    // generate targets
    let mut targets: Vec<Target> = Vec::new();
    let mut old_pos = [1600./40.; 2];
    for _ in 0..target_amt {
        let t = Target::new(old_pos, &mut rng);
        old_pos = t.pos;
        targets.push(t);
    }

    // main loop
    while t_time < t_thresh {
        // update drones
        for d in &mut *drones {
            // skip if done
            if d.crash || d.done {
                continue
            }

            // get the current Target
            let t: &Target = &targets[d.completed as usize];

            // update drone
            d.process(&t.pos);
            d.update(dt);
            d.survived += dt;

            // score drone
            score(d, t, dt);

            // check for completion
            if d.completed == target_amt {
                d.done = true;
            }
        }
        // time step
        t_time += dt;
    }

    // finalize scores
    for d in drones {
        if !d.done {
            let t: &Target = &targets[d.completed as usize];

            let dist = euclid_distance(&[t.pos[0] - d.pos[0], t.pos[1] - d.pos[1]]);
            let points = (t.points - dist).max(0.) * 0.9 / d.completion_time;

            d.score += points;
        }
        if d.crash {
            d.score /= 2.;
        } else if d.done {
            d.score *= (t_thresh + 1.) / (d.survived + 1.);
        }
    }
}
