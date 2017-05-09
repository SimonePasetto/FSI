import random
import numpy as np
import matplotlib.pyplot as plt

# Environment size
# from samba.dcerpc.epmapper.epm_entry_t import epm_entry_t

width = 5
height = 16

# Actions
num_actions = 4

actions_list = {"UP": 0,
                "RIGHT": 1,
                "DOWN": 2,
                "LEFT": 3
                }

actions_vectors = {"UP": (-1, 0),
                   "RIGHT": (0, 1),
                   "DOWN": (1, 0),
                   "LEFT": (0, -1)
                   }

# Discount factor
discount = 0.8

Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension

def getState(y, x):
    return y * width + x


def getStateCoord(state):
    return int(state / width), int(state % width)

def getActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions

def getActionsNumber(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append(1)
    if x > 0:
        actions.append(3)
    if y < height - 1:
        actions.append(2)
    if y > 0:
        actions.append(0)
    return actions

def movement(state, action):
    y = getStateCoord(state)[0] + actions_vectors[action][0]
    x = getStateCoord(state)[1] + actions_vectors[action][1]
    new_state = getState(y, x)
    return new_state

def getRndAction(state):
    return random.choice(getActions(state))

invert = {0: "UP",
          1: "RIGHT",
          2: "DOWN",
          3: "LEFT"}

def bestMove(state):
    possibleActions = getActionsNumber(state)
    action = possibleActions[Q[state, possibleActions].argmax()]
    action = invert[action]
    return action

def greedy(state):
    action = bestMove(state)
    new_state = movement(state, action)
    return new_state

def explore(state):
    action = getRndAction(state)
    new_state = movement(state, action)
    qlearning(state, actions_list[action], new_state)
    return new_state

def getRndState():
    return random.randint(0, height * width - 1)

Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000

Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return


# Episodes

# EXPLORE

pasos = 0

for i in xrange(100):
    state = getRndState()
    while state != final_state:
        pasos += 1
        state = explore(state)

print "\nNumero pasos (RANDOM): " , pasos/100
print "-----------------------------------------------------------------------"

# GREEDY

# Reset variables
Q = np.zeros((height * width, num_actions))  # Q matrix
pasos = 0

for i in xrange(100):
    state = getRndState()
    while state != final_state:
        pasos += 1
        if np.max(Q[state]) <= 0:
            state = explore(state)
        else:
            state = greedy(state)

print "\nNumero pasos (GREEDY): ", pasos/100
print "-----------------------------------------------------------------------"

# EPSILON-GREEDY

# Reset variables

for k in xrange(1, 5):
    Q = np.zeros((height * width, num_actions))  # Q matrix
    pasos = 0

    for i in xrange(100):
        state = getRndState()
        while state != final_state:
            pasos += 1
            test = random.uniform(0, 1);
            if np.max(Q[state]) <= 0 or test <= 0.2*k:        # EPSILON = 0.2, 0.4, 0.6, 0.8
                state = explore(state)
            else:
                state = greedy(state)

    print "\nNumero pasos (EPSILON-GREEDY): ", pasos/100
    print "Epsilon: ", 0.2*k


# Q matrix plot

s = 0
ax = plt.axes()
ax.axis([-1, width + 1, -1, height + 1])

for j in xrange(height):

    plt.plot([0, width], [j, j], 'b')
    for i in xrange(width):
        plt.plot([i, i], [0, height], 'b')

        direction = np.argmax(Q[s])
        if s != final_state:
            if direction == 0:
                ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 1:
                ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 2:
                ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 3:
                ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
        s += 1

    plt.plot([i+1, i+1], [0, height], 'b')
    plt.plot([0, width], [j+1, j+1], 'b')

plt.show()
