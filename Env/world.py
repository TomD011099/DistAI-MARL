import math
import random

from Env.predator import Predator
from Env.prey import Prey
from Env.simulator import Simulator
from Env.stats import Stats


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
        self.dead_prey = []
        self.dead_predators = []

        self.done = False

        for _ in range(pred_amount):
            self.predator_list.append(Predator(self, self.map_size, self.pred_id_cnt, self.pred_settings))
            self.pred_id_cnt += 1
        for _ in range(prey_amount):
            self.prey_list.append(Prey(self, self.map_size, self.prey_id_cnt, self.prey_settings))
            self.prey_id_cnt += 1

        self.stats = Stats(prey_amount, pred_amount)

        # self.simulator = Simulator(map_size)

    def step(self, actions=None, env_type=None):
        self.t += 1
        new_prey = []
        new_predators = []
        self.dead_prey.clear()
        self.dead_predators.clear()

        for p in self.prey_list:
            if env_type == "prey" or env_type == "multi":
                name = "prey_" + str(p.id)
                res = p.step(actions[name])
            else:
                res = p.step()

            if res[0][0]:
                new_prey.append(res[0][1])
            if res[1]:
                self.dead_prey.append(p)
        for p in new_prey:
            self.new_prey(p)

        for p in self.predator_list:
            if env_type == "pred" or env_type == "multi":
                name = "pred_" + str(p.id)
                res = p.step(actions[name])
            else:
                res = p.step()

            if res[0][0]:
                if res[0][1] not in self.dead_prey:
                    self.dead_prey.append(res[0][1])
            if res[1][0]:
                new_predators.append(res[1][1])
            if res[2]:
                self.dead_predators.append(p)

        for p in new_predators:
            self.new_pred(p)

        for p in self.dead_prey:
            self.del_prey(p)
        for p in self.dead_predators:
            self.del_pred(p)

        self.done = (len(self.predator_list) == 0 or len(self.prey_list) == 0) or self.t >= self.max_t
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

    def get_obs(self):
        out = self.get_pred_obs()
        out.update(self.get_prey_obs())

        return out

    def get_pred_obs(self):
        out = {}
        for p in self.predator_list:
            name = "pred_" + str(p.id)
            closest = self.closest_prey(p)
            if not closest:
                out[name] = [p.age, p.en_lvl, 0, 0]
            else:
                out[name] = [p.age, p.en_lvl, closest.pos[0] - p.pos[0], closest.pos[1] - p.pos[1]]
        for p in self.dead_predators:
            name = "pred_" + str(p.id)
            out[name] = [0, 0, 0, 0]
        return out

    def get_prey_obs(self):
        out = {}
        for p in self.prey_list:
            name = "prey_" + str(p.id)
            closest = self.closest_pred(p)
            if not closest:
                out[name] = [p.age, 0, 0]
            else:
                out[name] = [p.age, closest.pos[0] - p.pos[0], closest.pos[1] - p.pos[1]]
        for p in self.dead_prey:
            name = "prey_" + str(p.id)
            out[name] = [0, 0, 0]
        return out

    def get_rewards(self):
        out = {}
        for p in self.predator_list:
            name = "pred_" + str(p.id)
            out[name] = len(self.predator_list)
        for p in self.dead_predators:
            name = "pred_" + str(p.id)
            out[name] = 0
        for p in self.prey_list:
            name = "prey_" + str(p.id)
            out[name] = len(self.prey_list)
        for p in self.dead_prey:
            name = "prey_" + str(p.id)
            out[name] = 0

        return out

    def get_pred_rewards(self):
        out = {}
        for p in self.predator_list:
            name = "pred_" + str(p.id)
            out[name] = len(self.predator_list)
        for p in self.dead_predators:
            name = "pred_" + str(p.id)
            out[name] = 0
        return out

    def get_prey_rewards(self):
        out = {}
        for p in self.prey_list:
            name = "prey_" + str(p.id)
            out[name] = len(self.prey_list)
        return out

    def get_dones(self):
        out = {"__all__": self.done}
        for p in self.dead_predators:
            name = "pred_" + str(p.id)
            out[name] = True

        for p in self.dead_prey:
            name = "prey_" + str(p.id)
            out[name] = True

        return out

    def get_pred_dones(self):
        out = {"__all__": self.done}
        for p in self.dead_predators:
            name = "pred_" + str(p.id)
            out[name] = True
        return out

    def get_prey_dones(self):
        return {"__all__": self.done}

    def render(self):
        self.stats.update(len(self.prey_list), len(self.predator_list), self.t)
        self.simulator.update(self.predator_list, self.prey_list)
