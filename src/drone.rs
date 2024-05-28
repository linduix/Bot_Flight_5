use crate::neuralnet::NeuralNet;
use serde_json::Value;

fn vec_rotate(vector: &[f64; 2], rads: f64) -> [f64; 2] {
    let x = vector[0];
    let y = vector[1];
    let final_x = rads.cos() * x - rads.sin() * y;
    let final_y = rads.sin() * x + rads.cos() * y;

    return [final_x, final_y]
}

pub struct Drone<'a> {
    pub pos: [f64; 2],
    mass: f64,
    angle: f64,
    pub brain: NeuralNet<'a>,

    thrust_output: f32,
    angle_output: f32,
    max_thrust: f64,

    acceleration: [f64; 2],
    velocity: [f64; 2],

    pub completed: u32,
    pub score: f64,
    pub done: bool,
    pub crash: bool,

    pub completion_time: f64,
    pub touch_time: f64,
    pub survived: f64,
}

impl<'a> Drone<'a> {
    pub fn new(start_pos: [f64; 2], genotype: &Value) -> Drone {
        Drone {
            pos: start_pos,
            mass: 1.,
            angle: 0.,
            brain: NeuralNet::new(genotype),

            thrust_output: 0.,
            angle_output: 0.,
            max_thrust: 20.,

            acceleration: [0., 0.],
            velocity: [0., 0.],

            completed: 0,
            score: 0.,
            done: false,
            crash: false,

            completion_time: 0.,
            touch_time: 0.,
            survived: 0.,
        }
    }

    pub fn process(&mut self, target: &[f64; 2]) {
        let diffx = target[0] - self.pos[0];
        let diffy = target[1] - self.pos[1];
        let velx = self.velocity[0];
        let vely = self.velocity[1];
        let accx = self.acceleration[0];
        let accy = self.acceleration[1];
        let ang_s = self.angle.sin();
        let ang_c = self.angle.cos();

        let inp: [f64; 8] = [diffx, diffy, velx, vely, accx, accy, ang_s, ang_c];
        let output = self.brain.forward(&inp);

        let thrust = (output[0] + 1.)/2.;
        let angle = (output[1] + 1.)/2.;

        self.thrust_output = thrust as f32;
        self.angle_output = angle as f32;
    }

    fn rotate(&mut self, amt: f32, dt: f64) {
        let normalized = (amt - 0.5) as f64;
        let angle = normalized * 3. * std::f64::consts::PI * dt;
        self.angle = (self.angle + angle) % (2. * std::f64::consts::PI);
    }

    pub fn update(&mut self, dt: f64) {
        if self.done | self.crash {
            return;
        }

        // Rotate drone
        self.rotate(self.angle_output, dt);

        // Force calculations
        let force = [0., self.thrust_output as f64 * self.max_thrust];
        let rotated_force = vec_rotate(&force, self.angle);

        // Physics update
        self.acceleration[0] = rotated_force[0] / self.mass;
        self.acceleration[1] = rotated_force[1] / self.mass - 9.81;

        self.velocity[0] += self.acceleration[0] * dt;
        self.velocity[1] += self.acceleration[1] * dt;

        self.pos[0] += self.velocity[0] * dt;
        self.pos[1] += self.velocity[1] * dt;
    }
}