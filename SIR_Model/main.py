import numpy as np
import matplotlib.pyplot as plt

class SIRModel:
    def __init__(self, beta, gamma, S0, I0, R0):
        self.beta = beta
        self.gamma = gamma
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.N = S0 + I0 + R0  #total population

    def simulate(self, T, dt):
        num_steps = int(T / dt)

        #init empty arrays to store the results
        S = np.zeros(num_steps)
        I = np.zeros(num_steps)
        R = np.zeros(num_steps)
        time = np.zeros(num_steps)

        S[0] = self.S0
        I[0] = self.I0
        R[0] = self.R0

        for t in range(1, num_steps):
            # Euler method
            #NOTE added "/N" to the formula cause the original formula makes the population grow exponentially, 
            # now the diferential equations are normalized to the total population
            dS = -self.beta * S[t-1] * I[t-1] / self.N * dt
            dI = (self.beta * S[t-1] * I[t-1] / self.N - self.gamma * I[t-1]) * dt
            dR = self.gamma * I[t-1] * dt

            #apply the differentials
            S[t] = S[t-1] + dS
            I[t] = I[t-1] + dI
            R[t] = R[t-1] + dR

            #Ensure the variables are non-negative
            S[t] = max(S[t], 0)
            I[t] = max(I[t], 0)
            R[t] = max(R[t], 0)

            #simalate flow of time
            time[t] = t * dt

        return time, S, I, R
    
    def plot(self, time, S, I, R):
        plt.figure(figsize=(10,6))
        plt.plot(time, S, label='Susceptible')
        plt.plot(time, I, label='Infected')
        plt.plot(time, R, label='Recovered')
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.title("SIR Model Simulation with Euler's method")
        plt.legend()
        plt.grid(True)
        plt.show() 


"""
    Parameters
"""
beta = 0.3
gamma = 0.1

S0 = 990
I0 = 10
R0 = 0

T = 50
dt = 0.1

if __name__ == '__main__':
    sir = SIRModel(beta, gamma, S0, I0, R0)
    time, S, I, R = sir.simulate(T, dt)
    print(f"""
    RESULTS ON DAY {T}:
    - Susceptible: {round(S[-1])}
    - Infected: {round(I[-1])}
    - Recovered: {round(R[-1])}
    - Total Population: {round(S[-1] + I[-1] + R[-1])}
    """)
    sir.plot(time, S, I, R)