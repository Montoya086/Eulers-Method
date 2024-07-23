import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


class PopulationGrowth:
    def __init__(self, x0, y0, h, x_end, y_end, N):
        self.x0 = x0
        self.y0 = y0
        self.h = h
        self.x_end = x_end
        self.y_end = y_end
        self.N = N
        
    def growth_rate(self, xf, yf):
        self.k = fsolve(
            lambda k: yf - (
                self.N / (
                    1 + (self.N / self.y0 - 1) * np.exp(-k * (xf - self.x0))
                )
            ), 0)[0]
        
    def simulate(self):
        x = [self.x0]
        y = [self.y0]
        time_to_y_end = None
        
        while x[-1] + self.h <= self.x_end:
            
            if time_to_y_end is None and y[-1] >= self.y_end:
                time_to_y_end = x[-1]
            
            m = self.k * (1 - y[-1] / self.N) * y[-1]
            x.append(x[-1] + self.h)
            y.append(y[-1] + m * self.h)
            
        self.x = x
        self.y = y
        return x, y, time_to_y_end

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.y, label='Population')
        plt.xlabel('Years')
        plt.ylabel('Population')
        plt.title("Population Growth Simulation")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    x0 = 0
    y0 = 15
    x4 = 4
    y4 = 56
    h = 0.1
    x_end = 12
    y_end = 150
    N = 300
    
    model = PopulationGrowth(x0, y0, h, x_end, y_end, N)
    model.growth_rate(x4, y4)
    x, y, time_to_y_end = model.simulate()
    model.plot()
    
    print("En 12 días la población es de: {}".format(y[-1]))
    
    print("La población alcanza 150 mariposas en el día: {}"
          .format(time_to_y_end))
