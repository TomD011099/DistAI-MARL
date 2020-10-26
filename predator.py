import random

from gym import spaces


class Predator:
    def __init__(self, world, map_size, pred_id, pred_settings, pos=None):
        self.world = world
        self.age = 0
        self.map_size = map_size
        self.id = pred_id
        if pos is None:
            self.pos = (random.randint(0, map_size[0] - 1), random.randint(0, map_size[1] - 1))
        else:
            self.pos = pos
        self.pred_settings = pred_settings
        self.max_age = pred_settings[0]
        self.en_lvl = pred_settings[1]
        self.en_tr = pred_settings[2]
        self.en_ppe = pred_settings[3]
        if pred_settings[4] >= self.en_tr:
            self.en_thr = pred_settings[4]
        else:
            self.en_thr = self.en_tr

    def rand_action(self):
        return self.act(random.choice(range(5)))

    def act(self, action):
        dx, dy = 0, 0

        if action == 0:
            dx, dy = 0, 1
        elif action == 1:
            dx, dy = 1, 0
        elif action == 2:
            dx, dy = 0, -1
        elif action == 3:
            dx, dy = -1, 0
        elif action == 4:
            return self.reproduce()

        new_pos = (self.pos[0] + dx, self.pos[1] + dy)
        if self.valid_pos(new_pos):
            self.pos = new_pos
        return False, None

    def valid_pos(self, pos):
        return (0 <= pos[0] < self.map_size[0]) and (0 <= pos[1] < self.map_size[1])

    def eat(self):
        res = self.world.close_food(self)
        if res is not None:
            self.en_lvl += self.en_ppe
            return True, res
        else:
            return False, None

    def reproduce(self):
        if self.en_lvl > self.en_thr:
            self.en_lvl -= self.en_tr
            return True, self.pos
        else:
            return False, None

    def will_die(self):
        if (self.age >= self.max_age) or (self.en_lvl <= 0):
            return True
        else:
            return False

    def step(self, action=None):
        self.age += 1
        self.en_lvl -= 1
        if action is None:
            out = self.rand_action()
        else:
            out = self.act(action)

        out = (self.eat(), out, self.will_die())
        return out
