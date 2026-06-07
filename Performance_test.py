# For the graph how things changing on changing the paramenters I will use only one agent because it will be similar for all the agents
import pygame
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

# Initialize pygame(starting the game engine)
#pygame.init()


# Color tuples
WHITE = (255, 255, 255)
GRAY = (20, 20, 20) # 0.1 is the alpha value for transparency
BLUE = (50, 100, 220)
ORANGE = (255, 165, 0)
PURPLE = (160, 32, 240)
BLACK = (0, 0, 0)

# Action space
actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

def detect_collision(agent_positions, human_positions, x):
    agent_pos = agent_positions[x]
    # Other agents are all agents excluding the current one
    other_agent_positions = agent_positions[:x] + agent_positions[x+1:]
    
    # Check for collision with humans or other agents
    if agent_pos in human_positions:
        #print(f"Agent {x} collided with a human at position {agent_pos}")
        return True
    if agent_pos in other_agent_positions:
        #print(f"Agent {x} collided with another agent at position {agent_pos}")
        return True

    #print(f"No collision detected for agent {x}")
    return False
def train_agent(start_pos, goal, other_goals, agent_name):
    Q = init_Q()
    for episode in range(episodes):
        pos = start_pos
        while pos != goal:
            
            # exploration-explotation following epsilon-greedy policy
            if random.random() < epsilon:
                action = random.choice(list(actions))
            else:
                action = max(Q[pos], key=Q[pos].get)

            # now, we have assigned the action, let's execute the action and observe the reward accordingly
            next_pos, reward = take_action(pos, action, goal, other_goals)
            Q[pos][action] += alpha * (reward + gamma * max(Q[next_pos].values()) - Q[pos][action])
            pos = next_pos
    
    return Q

def move_human_randomly(pos):
    directions = list(actions.values())
    random.shuffle(directions) # It just suffles the order of actions => just like a card suffle  
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in goal_positions and grid[nx, ny] != OBSTACLE):
            return (nx, ny) # this is a valid move for the human
    return pos  # Stay in place if no valid move

def take_action(pos, action, goal, other_goals):
    x, y = pos
    dx, dy = actions[action]
    nx, ny = x + dx, y + dy

    if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
        return pos, -10  # out of bounds

    if grid[nx, ny] == OBSTACLE:
        return pos, -100  # static wall
    if (nx, ny) in other_goals:
        return pos, -50  # other agent’s goal treated as wall

    if (nx, ny) == goal:
        return (nx, ny), 100  # success
    return (nx, ny), -1  # regular step



# this function takes current position, action assigned, goal position and other goals positions as input and returns the 'next position' and 'reward' based on the action taken used to update Q-values.

# Moniter Progress Visualization
def draw_progress_bar(screen, episode, total_episodes, agent_name, width=400, height=30):
    progress = episode / total_episodes
    bar_width = int(width * progress)

    x = (WIDTH - width) // 2
    y = (HEIGHT - height) // 2

    pygame.draw.rect(screen, (230, 230, 230), (x, y, width, height))
    pygame.draw.rect(screen, (0, 180, 0), (x, y, bar_width, height))

    font = pygame.font.SysFont(None, 28)
    text = font.render(f'Training {agent_name} - {int(progress * 100)}%', True, (0, 0, 0))
    text_rect = text.get_rect(center=(WIDTH // 2, y - 25))
    screen.blit(text, text_rect)

    pygame.display.flip()

# Let's initialize all Q values to 0 for all the cells
def init_Q():
    return {(i, j): {a: 0.0 for a in actions} for i in range(GRID_SIZE) for j in range(GRID_SIZE)}

#*****************************************************************************************************************************
# Changable parameters
x=0 #number_of_humans= 50
a=25 #GRID_SIZE = 25
b=24 #CELL_SIZE = int(24 * 1.2)
c=-1 #OBSTACLE = -1 #obstacle reward
d=100 #GOAL = 10 #goal reward
goal_positions = [(19, 7), (19, 10), (10, 13)]
K=0.3 #ALPHA
L=0.9 #GAMMA
M=0.4 #EPSILON
N = 1000 #episodes
p=0.2

#Let's devide the episodes in this manner: 100, 200,300 upto 100000
Discrete_episodes = []
WW=280 # If code stucks then this starting point can be insufficient 
while WW< N:
    if WW <600: #because initially changes are more effective and visible
        Discrete_episodes.append(WW)
        WW+=30
    else:
        M=0.2 #EPSILON
        WW+=400
        Discrete_episodes.append(WW)
    
print(len(Discrete_episodes), "discrete episodes are generated for graphing purpose")

no_of_steps=[]
cumulated_reward=[]

for N_episoeds in Discrete_episodes:
    # set the game display
    GRID_SIZE = a
    CELL_SIZE = b # this is one cell size ca be adjusted for visibility accordingly
    WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

    # OBSTACLE AND GOAL
    OBSTACLE = c #obstacle reward
    GOAL = d #goal reward
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int) #used for grid representation
    grid[5, 7:18] = OBSTACLE # Static Obstacles(brics)
    grid[8, 18:25] = OBSTACLE
    grid[15, 6:14] = OBSTACLE
    grid[15, 20:25] = OBSTACLE
    grid[10, 0:11] = OBSTACLE
    grid[8:12, 16] = OBSTACLE
    grid[20, 6:16] = OBSTACLE

    # Goal positions
    goal_positions = [(19, 7), (19, 10), (10, 13)]
    goal_colors = [BLUE, ORANGE, PURPLE]
    for goal in goal_positions:
          grid[goal] = GOAL

    # Q-learning parameters
    alpha, gamma, epsilon = K, L, M

    episodes = N_episoeds # The more number of episodes, the more our agent explores the envirement (I must keep it bigger when envirement has more obstacles)

    percentage_of_pause = p # during collision pause a robot for 50% and rest time let others move
    Q1 = train_agent((0, 0), goal_positions[0], goal_positions[1:], "Agent 1")
    Q2 = train_agent((0, GRID_SIZE-1), goal_positions[1], [goal_positions[0], goal_positions[2]], "Agent 2")
    Q3 = train_agent((GRID_SIZE-1, GRID_SIZE-1 ), goal_positions[2], goal_positions[:2], "Agent 3")
    print("Finished ", N_episoeds)


    # Initialize moving humans at random positions
    number_of_humans= x
    human_positions = []
    while len(human_positions) < number_of_humans:
        pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        if grid[pos] == 0 and pos not in goal_positions and pos not in [(0, 0), (0, GRID_SIZE - 1), (GRID_SIZE - 1, GRID_SIZE - 1)] and pos not in human_positions:
            human_positions.append(pos)

    # Let's selsct a simple policy: agent will select only that action that will yield to max Q-value for that state(greedy policy)
    policy1 = {s: max(Q1[s], key=Q1[s].get) for s in Q1} 
    cumulated_reward1= sum(Q1[s][policy1[s]] for s in Q1) 
    
    cumulated_reward.append(cumulated_reward1)
    
    policy2 = {s: max(Q2[s], key=Q2[s].get) for s in Q2}
    cumulated_reward2 = sum(Q2[s][policy2[s]] for s in Q2) 

    policy3 = {s: max(Q3[s], key=Q3[s].get) for s in Q3}
    cumulated_reward3 = sum(Q3[s][policy3[s]] for s in Q3) 


    total_steps= 0
    def following_policy_and_reaching_goal():
        start_positions = [(0, 0), (0, GRID_SIZE-1), (GRID_SIZE-1,GRID_SIZE-1)]
        agent_positions = list(start_positions)
        paths = [[pos] for pos in agent_positions] # list of lists to store the path of each agent
        policies = [policy1, policy2, policy3]
        goals = goal_positions
        running = True
    
        while running:
            global human_positions
            human_positions = [move_human_randomly(pos) for pos in human_positions]
            previous_positions = agent_positions.copy() 
            for i in range(len(agent_positions)):
                if agent_positions[i] != goals[i]:
                    action = policies[i][agent_positions[i]]
                    other_goals = [g for idx, g in enumerate(goals) if idx != i]
                    next_pos, _ = take_action(agent_positions[i], action, goals[i], other_goals)
                    agent_positions[i] = next_pos 
        
    
            for i in range(len(agent_positions)):
                if detect_collision(agent_positions, human_positions, i):
                    if random.random() < percentage_of_pause:
                        agent_positions[i] = previous_positions[i]
                    else:
                        # Find a safe position to move
                        found_safe_position = False
                        for dx, dy in actions.values():
                            nx, ny = previous_positions[i][0] + dx, previous_positions[i][1] + dy
                            if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 
                                grid[nx, ny] != OBSTACLE and  
                                (nx, ny) not in human_positions and 
                                (nx, ny) not in agent_positions):
                                agent_positions[i] = (nx, ny)
                                paths[i].append(agent_positions[i])
                                found_safe_position = True
                                break
                    
                        if not found_safe_position:
                            agent_positions[i] = previous_positions[i]  # it means no action taken
                            paths[i].append(agent_positions[i])
                            
                else:
                    #no collision, so let's just append the current position to the path for drawing
                    paths[i].append(agent_positions[i])
            
    
            # Vallah we have reached then end the simulation (the main loop)
            if all(agent_positions[i] == goals[i] for i in range(len(agent_positions))):
                running = False
        total_steps = sum(len(path) for path in paths)
        no_of_steps.append(total_steps)
    
    following_policy_and_reaching_goal()
    # LET'S reset Q-values for the next number of episodes

# Plotting no_of_steps vs Discrete_episodes
plt.figure(figsize=(10, 6))
plt.plot(Discrete_episodes, no_of_steps, marker='o', label='No of Steps')
plt.title('No Of Steps To Reach Goal vs Training_Episodes')
plt.xlabel('No Of Episodes Used For Training')
plt.ylabel('No of Steps')
plt.grid()
plt.legend()
plt.savefig('No_of_steps_to_reach_goal_vs_Training_episodes.png')

# Plotting cumulated_reward vs Discrete_episodes
plt.figure(figsize=(10, 6))
plt.plot(Discrete_episodes, cumulated_reward, marker='o', label='Cumulated Reward')
plt.title('Cumulated Reward vs Training_episodes')
plt.xlabel('No Of Episodes Used For Training')
plt.ylabel('Cumulated Reward')
plt.grid()
plt.legend()
plt.savefig('Cumulated_Reward_vs_Training_episodes.png')
plt.show()


no_of_steps=[]
L1,L2=1,100
episodes = 10000
percentage_of_pause = 0.0
#For Steps vs objects
for xx in range(L1,L2):
    # set the game display
    GRID_SIZE = a
    CELL_SIZE = b # this is one cell size ca be adjusted for visibility accordingly
    WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

    # OBSTACLE AND GOAL
    OBSTACLE = c #obstacle reward
    GOAL = d #goal reward
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int) #used for grid representation
    grid[5, 7:18] = OBSTACLE # Static Obstacles(brics)
    grid[8, 18:25] = OBSTACLE
    grid[15, 6:14] = OBSTACLE
    grid[15, 20:25] = OBSTACLE
    grid[10, 0:11] = OBSTACLE
    grid[8:12, 16] = OBSTACLE
    grid[20, 6:16] = OBSTACLE

    # Goal positions
    goal_positions = [(19, 7), (19, 10), (10, 13)]
    goal_colors = [BLUE, ORANGE, PURPLE]
    for goal in goal_positions:
          grid[goal] = GOAL

    # Q-learning parameters
    alpha, gamma, epsilon = K, L, M
    Q1 = train_agent((0, 0), goal_positions[0], goal_positions[1:], "Agent 1")
    Q2 = train_agent((0, GRID_SIZE-1), goal_positions[1], [goal_positions[0], goal_positions[2]], "Agent 2")
    Q3 = train_agent((GRID_SIZE-1, GRID_SIZE-1 ), goal_positions[2], goal_positions[:2], "Agent 3")
    print("Processing Total Moving Obstacles= ",xx)


    # Initialize moving humans at random positions
    number_of_humans= xx
    human_positions = []
    while len(human_positions) < number_of_humans:
        pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        if grid[pos] == 0 and pos not in goal_positions and pos not in [(0, 0), (0, GRID_SIZE - 1), (GRID_SIZE - 1, GRID_SIZE - 1)] and pos not in human_positions:
            human_positions.append(pos)

    # Let's selsct a simple policy: agent will select only that action that will yield to max Q-value for that state(greedy policy)
    policy1 = {s: max(Q1[s], key=Q1[s].get) for s in Q1} 
    cumulated_reward1= sum(Q1[s][policy1[s]] for s in Q1) 
    
    cumulated_reward.append(cumulated_reward1)
    
    policy2 = {s: max(Q2[s], key=Q2[s].get) for s in Q2}
    cumulated_reward2 = sum(Q2[s][policy2[s]] for s in Q2) 

    policy3 = {s: max(Q3[s], key=Q3[s].get) for s in Q3}
    cumulated_reward3 = sum(Q3[s][policy3[s]] for s in Q3) 


    total_steps= 0
    def following_policy_and_reaching_goal():
        start_positions = [(0, 0), (0, GRID_SIZE-1), (GRID_SIZE-1,GRID_SIZE-1)]
        agent_positions = list(start_positions)
        paths = [[pos] for pos in agent_positions] # list of lists to store the path of each agent
        policies = [policy1, policy2, policy3]
        goals = goal_positions
        running = True
    
        while running:
            global human_positions
            human_positions = [move_human_randomly(pos) for pos in human_positions]
            previous_positions = agent_positions.copy() 
            for i in range(len(agent_positions)):
                if agent_positions[i] != goals[i]:
                    action = policies[i][agent_positions[i]]
                    other_goals = [g for idx, g in enumerate(goals) if idx != i]
                    next_pos, _ = take_action(agent_positions[i], action, goals[i], other_goals)
                    agent_positions[i] = next_pos 
        
    
            for i in range(len(agent_positions)):
                if detect_collision(agent_positions, human_positions, i):
                    if random.random() < percentage_of_pause:
                        agent_positions[i] = previous_positions[i]
                    else:
                        # Find a safe position to move
                        found_safe_position = False
                        for dx, dy in actions.values():
                            nx, ny = previous_positions[i][0] + dx, previous_positions[i][1] + dy
                            if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 
                                grid[nx, ny] != OBSTACLE and  
                                (nx, ny) not in human_positions and 
                                (nx, ny) not in agent_positions):
                                agent_positions[i] = (nx, ny)
                                paths[i].append(agent_positions[i])
                                found_safe_position = True
                                break
                    
                        if not found_safe_position:
                            agent_positions[i] = previous_positions[i]  # it means no action taken
                            paths[i].append(agent_positions[i])
                            
                else:
                    #no collision, so let's just append the current position to the path for drawing
                    paths[i].append(agent_positions[i])
            
    
            # Vallah we have reached then end the simulation (the main loop)
            if all(agent_positions[i] == goals[i] for i in range(len(agent_positions))):
                running = False
        total_steps = sum(len(path) for path in paths)
        no_of_steps.append(total_steps)
    
    following_policy_and_reaching_goal()
    # LET'S reset Q-values for the next number of episodes



Obstacles=[v for v in range(L1,L2)]
# Steps vs random_obstacles
plt.figure(figsize=(10, 6))
plt.plot(Obstacles, no_of_steps, marker='o', label='No of Steps')
plt.title('No Of Steps To Reach Goal vs Obstacles')
plt.xlabel('No Of Obstacles')
plt.ylabel('No of Steps')
plt.grid()
plt.legend()
plt.savefig('No_of_steps_to_reach_goal_vs_Training_episodes.png')
plt.show()
