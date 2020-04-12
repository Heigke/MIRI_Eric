import numpy as np
from gym import spaces
import optimal_lqr_control

class Automatic_Control_Environment():
    """ ***A simle automatic control environment***
    by Niklas Kotarsky and Eric Bergvall
    
    The system is described by x_t+1 = A*x_t + B*u_t + C*noise
    where x_t is a column vector with dimension N and A has dimension N x N
    u_t has dimension M and B then have dimension NxM 
    Noise has dimension N and C has dimension NxN """
    def __init__(self,A,B,Q,R,N,initial_value,C=0,horizon=100):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.N = N
        self.horizon = horizon
        self.initial_value = initial_value
        self.state = self.initial_value
        self.initial_action = np.random.normal(0,1,(self.B.shape[1],1))
        self.action = self.initial_action
        self.state_limit = 1000
        self.nbr_steps = 0
        self.high = 10
        self.action_space = spaces.Box(low=-self.high, high=self.high, shape=self.action.shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.high, high=self.high, shape=self.state.shape, dtype=np.float32)

        self.lqr_optimal = optimal_lqr_control.Lqr(A,B,Q,R,N,horizon)
        

    def state_space_equation(self, action):
        noise = np.random.normal(0,1,self.state.shape)
        new_state = self.A@self.state+self.B@action+self.C*noise
        # Y är RLs state = C@self.state 
        return new_state

    def step(self, action):
        next_state = self.state_space_equation(action)
        done = self.done()
        self.state = next_state
        self.action = action
        _ = []
        reward = self.reward()
        self.nbr_steps += 1
        next_state = next_state.squeeze()
        return next_state, reward, done, _
    def render(self, mode='human'):
        nonsense=1
        return
    def close(self):
        nonsense=1
        return
    def reset(self):
        self.state = self.initial_value
        self.action = self.initial_action
        self.nbr_steps = 0
        self.lqr_optimal.reset()
        return self.state

    def _get_obs(self):
        return self.state

    def reward(self):
        x = self.state
        u = self.action
        x_T = np.transpose(x)
        u_T = np.transpose(u)
        Q = self.Q
        R = self.R
        N = self.N
        current_reward = x_T@Q@x+u_T@R@u+2*x_T@N@x
        return -current_reward

    def done(self):
            return False


if __name__ == "__main__":
    A = np.array([[1,0],[0,1]])
    B = np.array([[1,0],[0,1]])
    Q = np.array([[1,0],[0,1]])
    R = np.array([[1,0],[0,1]])
    N = np.array([[0,0],[0,0]])
    initial_value = np.array([[1],[1]])
    ac_env = Automatic_Control_Environment(A,B,Q,R,N,initial_value)
    print("obs space: "+str(ac_env.observation_space.shape))
    print("act space: "+str(ac_env.action_space.shape))
    state = ac_env.reset()
    optimal_action = ac_env.lqr_optimal.action(initial_value)
    action = np.array([[5],[5]])
    next_state, reward, done, _ = ac_env.step(action)
    print("new state")
    print(next_state.)
    print("rew")
    print(reward)
    print(done)
    next_state, reward, done, _ = ac_env.step(action)
    print("new state")
    print(next_state)
    print("rew")
    print(reward)
    print(done)
    print(state)
    print(state.shape)