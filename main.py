import time

from world import World

if __name__ == "__main__":
    # Prey settings:
    # birth_rate = prey_settings[0]
    # max_age = prey_settings[1]

    # Pred settings:
    # max_age = pred_settings[0]
    # en_lvl = pred_settings[1]
    # en_tr = pred_settings[2]
    # en_ppe = pred_settings[3]
    # en_thr = pred_settings[4]
    world = World((20, 20), (17, 6), 100, (20, 20, 30, 10, 40), 20, 500)

    stop = False
    while not stop:
        stop = world.step()
        time.sleep(0.1)

    world.stats.show_hist()
