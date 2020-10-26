from matplotlib import pyplot as plt


class Stats:
    def __init__(self, prey_init, pred_init):
        self.prey_hist = []
        self.prey_hist.append(prey_init)
        self.pred_hist = []
        self.pred_hist.append(pred_init)
        self.t = []
        self.t.append(0)

    def update(self, prey_amount, pred_amount, t):
        self.prey_hist.append(prey_amount)
        self.pred_hist.append(pred_amount)
        self.t.append(t)
        print("t:", t, "\t - Prey:", prey_amount, "\t - Pred:", pred_amount)

    def show_hist(self):
        plt.title("Stats")
        plt.xlabel("Time")
        plt.ylabel("Amount")
        plt.plot(self.t, self.prey_hist, label="Prey")
        plt.plot(self.t, self.pred_hist, label="Predators")
        plt.legend()
        plt.show()
