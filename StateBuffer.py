import random
import math

class StateMem:
    
    def __init__(self, x, y, theta, vx, vy, w, action):
        self.x = x
        self.y = y
        self.theta = theta
        self.vx = vx
        self.vy = vy
        self.w = w
        self.action = action

    def __repr__(self):
        return f'StateMem("{self.x}", {self.y}, {self.theta}, {self.vx}, {self.vy}, {self.w}, {self.action})'

    def values(self):
        return self.x, self.y, self.theta, self.vx, self.vy, self.w, self.action 

class StateBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.states = []

    def append(self, stateMem):
        self.states.append(stateMem)

        if len(self.states) > self.max_size:
            del self.states[0:len(self.states)-self.max_size]

    def extend(self, stateBuffer):
        self.states.extend(stateBuffer.states)

        if len(self.states) > self.max_size:
            del self.states[0:len(self.states)-self.max_size]

    def sample(self, delete_flag=False):
        index = random.randint(0, len(self.states)-1)
        if delete_flag:
            return self.states.pop(index)
        else:
            return self.states[index]

    def generate(self, xs, ys, thetas, vx=0, vy=0, w=0):
        for x in xs:
            for y in ys:
                for theta in thetas:
                    stateMem = StateMem(x, y, theta, vx, vy, w, [0,0,0])
                    self.states.append(stateMem)
                    
    def generate2(self, xs, ys, thetas, vx=0, vy=0, w=0):
        for x in xs:
            for y in ys:
                for theta in thetas:
                    stateMem = StateMem(x, y, theta, vx, vy, w, [1.0, 0, 0])
                    self.states.append(stateMem)
                    stateMem = StateMem(x, y, theta, vx, vy, w, [0, 0, math.pi/4])
                    self.states.append(stateMem)

    def generate3(self, xs, ys, thetas, vx=0, vy=0, w=0):
        for x in xs:
            for y in ys:
                for theta in thetas:
                    stateMem = StateMem(x, y, theta, vx, vy, w, [0,0,0])
                    self.states.append(stateMem)
                    stateMem = StateMem(x, y, theta, vx, vy, w, [1.0, 0, 0])
                    self.states.append(stateMem)
                    stateMem = StateMem(x, y, theta, vx, vy, w, [0, 0, math.pi/4])
                    self.states.append(stateMem)
                    
    def size(self):
        return len(self.states)

if __name__ == '__main__':

    import random
    import math

    stateBuffer = StateBuffer(max_size = 10000)
    xs = [-3.0, -1.5, 0.0, 1.5, 3.0]
    ys = [-3.0, 0.0, 3.0]

    p4 = math.pi / 4.0
    thetas = [0.0, p4, 2.0*p4, 3.0*p4, 4.0*p4, 5.0*p4, 6.0*p4, 7.0*p4]
    stateBuffer.generate(xs, ys, thetas)

    from pprint import pprint
    # pprint(stateBuffer.states)
    # print(len(stateBuffer.states))

    while stateBuffer.size() > 0:
        pprint(stateBuffer.sample(delete_flag=True))
        # print(len(stateBuffer.states))



