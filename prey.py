import random

from gym import spaces


class Prey:
    def __init__(self, world, map_size, prey_id, prey_settings, pos=None):
        self.world = world
        self.age = 0
        self.map_size = map_size
        self.id = prey_id
        if pos is None:
            self.pos = (random.randint(0, map_size[0]), random.randint(0, map_size[1]))
        else:
            self.pos = pos
        self.prey_settings = prey_settings
        self.birth_rate = prey_settings[0]
        self.max_age = prey_settings[1]

    def move_rand(self):
        self.move(random.choice(range(4)))

    def move(self, move):
        dx, dy = 0, 0

        if move == 0:
            dx, dy = 0, 1
        elif move == 1:
            dx, dy = 1, 0
        elif move == 2:
            dx, dy = 0, -1
        elif move == 3:
            dx, dy = -1, 0

        new_pos = (self.pos[0] + dx, self.pos[1] + dy)
        if self.valid_pos(new_pos):
            self.pos = new_pos
            return True
        else:
            return False

    def valid_pos(self, pos):
        return (0 <= pos[0] < self.map_size[0]) and (0 <= pos[1] < self.map_size[1])

    def reproduce(self):
        chance = random.randint(1, 100)
        if chance < self.birth_rate:
            return True, self.pos
        else:
            return False, None

    def will_die(self):
        if self.age >= self.max_age:
            return True
        else:
            return False

    def step(self, action=None):
        self.age += 1
        if action is None:
            self.move_rand()
        else:
            self.move(action)
        out = self.reproduce()
        out = out, self.will_die()
        return out
