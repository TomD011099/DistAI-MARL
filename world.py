import random
import math
import numpy as np

from prey import Prey
from predator import Predator
from simulator import Simulator
from stats import Stats


class World:
    def __init__(self, map_size, prey_settings, prey_amount, pred_settings, pred_amount, max_t):
        self.t = 0
        self.max_t = max_t
        self.map_size = map_size

        self.predator_list = []
        self.prey_list = []
        self.pred_settings = pred_settings
        self.prey_settings = prey_settings
        self.pred_id_cnt = 0
        self.prey_id_cnt = 0

        self.done = False

        for _ in range(pred_amount):
            self.predator_list.append(Predator(self, self.map_size, self.pred_id_cnt, self.pred_settings))
            self.pred_id_cnt += 1
        for _ in range(prey_amount):
            self.prey_list.append(Prey(self, self.map_size, self.prey_id_cnt, self.prey_settings))
            self.prey_id_cnt += 1

        self.stats = Stats(prey_amount, pred_amount)

        self.simulator = Simulator(map_size)

    def step(self, actions=None, env_type=None):
        self.t += 1
        new_prey = []
        new_predators = []
        dead_prey = []
        dead_predators = []

        for p in self.prey_list:
            if env_type == "prey":
                name = "prey_" + str(p.id)
                res = p.step(actions[name])
            else:
                res = p.step()

            if res[0][0]:
                new_prey.append(res[0][1])
            if res[1]:
                dead_prey.append(p)
        for p in new_prey:
            self.new_prey(p)

        for p in self.predator_list:
            if env_type == "pred":
                name = "pred_" + str(p.id)
                res = p.step(actions[name])
            else:
                res = p.step()

            if res[0][0]:
                if res[0][1] not in dead_prey:
                    dead_prey.append(res[0][1])
            if res[1][0]:
                new_predators.append(res[1][1])
            if res[2]:
                dead_predators.append(p)
        for p in new_predators:
            self.new_pred(p)

        for p in dead_prey:
            self.del_prey(p)
        for p in dead_predators:
            self.del_pred(p)

        self.stats.update(len(self.prey_list), len(self.predator_list), self.t)
        self.done = self.simulator.update(self.predator_list, self.prey_list) or (len(self.predator_list) == 0 and len(
            self.prey_list) == 0) or self.t >= self.max_t
        return self.done

    def close_food(self, pred):
        close = []

        for p in self.prey_list:
            if p.pos == pred.pos:
                close.append(p)

        if len(close) != 0:
            return random.choice(close)
        else:
            return None

    def closest_pred(self, prey):
        closest = []
        closest_distance = math.inf

        for p in self.predator_list:
            dx = prey.pos[0] - p.pos[0]
            dy = prey.pos[1] - p.pos[1]
            d = dx*dx + dy*dy
            if d < closest_distance:
                closest_distance = d
                closest = [p]
            elif d == closest_distance:
                closest.append(p)

        if len(closest) != 0:
            return random.choice(closest)
        else:
            return None

    def closest_prey(self, pred):
        closest = []
        closest_distance = math.inf

        for p in self.prey_list:
            dx = pred.pos[0] - p.pos[0]
            dy = pred.pos[1] - p.pos[1]
            d = dx * dx + dy * dy
            if d < closest_distance:
                closest_distance = d
                closest = [p]
            elif d == closest_distance:
                closest.append(p)

        if len(closest) != 0:
            return random.choice(closest)
        else:
            return None

    def new_pred(self, pos):
        self.predator_list.append(Predator(self, self.map_size, self.pred_id_cnt, self.pred_settings, pos))
        self.pred_id_cnt += 1

    def del_pred(self, pred):
        self.predator_list.remove(pred)

    def new_prey(self, pos):
        self.prey_list.append(Prey(self, self.map_size, self.prey_id_cnt, self.prey_settings, pos))
        self.prey_id_cnt += 1

    def del_prey(self, prey):
        self.prey_list.remove(prey)

    def get_pred_obs(self):
        out = np.array([])
        for p in self.predator_list:
            closest = self.closest_prey(p)
            np.append(out, [p.age, p.en_lvl, closest.pos[0] - p.pos[0], closest.pos[1] - p.pos[1]])
        return out

    def get_prey_obs(self):
        out = np.array([])
        for p in self.prey_list:
            closest = self.closest_pred(p)
            np.append(out, [p.age, closest.pos[0] - p.pos[0], closest.pos[1] - p.pos[1]])
        return out

    def get_pred_rewards(self):
        out = np.array([])
        for p in self.predator_list:
            # TODO implement
            i = 0
            np.append(out, i)
        return out

    def get_prey_rewards(self):
        out = np.array([])
        for p in self.prey_list:
            # TODO implement
            i = 0
            np.append(out, i)
        return out

    def get_pred_dones(self):
        if self.done:
            out = np.ones(len(self.predator_list), dtype=bool)
        else:
            out = np.zeros(len(self.predator_list), dtype=bool)
        return out

    def get_prey_dones(self):
        if self.done:
            out = np.ones(len(self.prey_list), dtype=bool)
        else:
            out = np.zeros(len(self.prey_list), dtype=bool)
        return out
