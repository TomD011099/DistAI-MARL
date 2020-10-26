import pygame
import copy
import math


class Simulator:
    def __init__(self, map_size):
        x_max, y_max = map_size

        x_zoom = math.floor(1920/x_max)
        y_zoom = math.floor(1080/y_max)
        self.zoom = min(x_zoom, y_zoom)

        pygame.init()
        size = x_max * self.zoom, y_max * self.zoom
        self.screen = pygame.display.set_mode(size)

        self.prey_img = pygame.image.load("img/Prey.png")
        self.prey_img = pygame.transform.scale(self.prey_img, (self.zoom, self.zoom))
        self.prey_rect = self.prey_img.get_rect()

        self.pred_img = pygame.image.load("img/Pred.png")
        self.pred_img = pygame.transform.scale(self.pred_img, (self.zoom, self.zoom))
        self.pred_rect = self.pred_img.get_rect()

    def update(self, predator_list, prey_list):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

        prey = []
        predators = []

        for p in predator_list:
            temp = copy.deepcopy(self.pred_rect)
            temp.x = p.pos[0] * self.zoom
            temp.y = p.pos[1] * self.zoom
            predators.append(temp)

        for p in prey_list:
            temp = copy.deepcopy(self.prey_rect)
            temp.x = p.pos[0] * self.zoom
            temp.y = p.pos[1] * self.zoom
            prey.append(temp)

        self.screen.fill((255, 255, 255))
        for i in predators:
            self.screen.blit(self.pred_img, i)
        for i in prey:
            self.screen.blit(self.prey_img, i)
        pygame.display.flip()

        return False
