import numpy as np
from pyfiles.NeuralNetwork import NeuralNetwork
import ast


class AiDrone(object):
    @staticmethod
    def _rotatemat(vec2d: np.ndarray, rads: float) -> np.ndarray:
        cos_rads = np.cos(rads)
        sin_rads = np.sin(rads)
        x, y = vec2d
        final_x = cos_rads * x - sin_rads * y
        final_y = sin_rads * x + cos_rads * y
        return np.array([final_x, final_y])

    def rotate(self, amt, dt):
        normalized = amt-0.5
        angle = normalized * 3*np.pi*dt
        self.angle = (self.angle + angle) % (2*np.pi)

    def __init__(self, startpos: list[2], genotype: dict = None):
        self.startpos = np.array(startpos, dtype=np.float32)
        self.pos: np.ndarray = self.startpos.copy()  # x, y
        self.old_pos: np.ndarray = self.pos.copy()
        self.mass: float = 1
        self.angle: float = 0  # rads: rotation is Coutner-clockwise from up
        if genotype:
            self.brain = NeuralNetwork(genotype=genotype)
        else:
            self.brain = NeuralNetwork()

        self.thrust_output: float = 0  # % output of the thrusters
        self.angle_output: float = 0  # rotation target
        self.MaxThrust: float = 20

        self.acc: np.ndarray = np.array([0., 0.])
        self.vel: np.ndarray = np.array([0., 0.])

        self.completed: int = 0  # number of completed targets
        self.score: float = 1  # total score of the drone: higher is better
        self.done: bool = False  # completion status
        self.crash: bool = False  # crash status

        self.completion_time: float = 0  # total time passed since new target
        self.touch_time: float = 0  # the time spent 'touching' the target
        self.survived: float = 0  # survival time

        self.species: int = None
        self.protection: int = 0

    def process(self, target: list[2]):
        target = np.array(target)
        diffx, diffy = target - self.pos
        velx, vely = self.vel
        accx, accy = self.acc
        angS, angC = np.sin(self.angle), np.cos(self.angle)

        inp = [diffx, diffy, velx, vely, accx, accy, angS, angC]
        output = self.brain(inp)

        normalized_output = (output[0]+1)/2, (output[1]+1)/2
        self.thrust_output, self.angle_output = normalized_output

    def update(self, dt):
        if self.done or self.crash:
            return

        # Rotate drone
        self.rotate(self.angle_output, dt)

        # Force calculations
        force = np.array([0, self.thrust_output*self.MaxThrust])
        angled_thrust = self._rotatemat(force, self.angle)

        # Physics updates
        self.acc = angled_thrust / self.mass + np.array([0., -9.81])  # Gravity
        self.vel += self.acc * dt

        self.old_pos = self.pos.copy()
        self.pos += self.vel * dt

    def reset(self):
        self.pos: np.ndarray = self.startpos.copy()
        self.angle: float = 0
        self.old_pos = self.pos.copy()

        self.angle_output: float = 0
        self.thrust_output: float = 0

        self.acc: np.ndarray = np.array([0., 0.])
        self.vel: np.ndarray = np.array([0., 0.])

        self.score: float = 0
        self.completed: int = 0

        self.touch_time: float = 0
        self.completion_time: float = 0
        self.survived: float = 0

        self.done: bool = False
        self.crash: bool = False

    @property
    def genotype(self):
        return self.brain.genotype

    @genotype.setter
    def genotype(self, genotype):
        self.brain.genotype = genotype.copy()


class DroneData(object):
    def __init__(self, genotype: dict, score, completed, crash, survived):
        # Format genotype
        genotype['connections'] = {ast.literal_eval(k): v for k, v in genotype['connections'].items()}
        genotype['nodes'] = {int(k): v for k, v in genotype['nodes'].items()}

        self.genotype = genotype
        self.score = score
        self.completed = completed
        self.crash = crash
        self.survived = survived
        self.species = None


if __name__ == '__main__':
    drone = AiDrone([0, 0])
    print(inp:=np.random.rand(8, 1))
    print(drone.brain(inp))