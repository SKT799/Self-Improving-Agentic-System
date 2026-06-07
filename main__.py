import pygame
import numpy as np
import random
import sys

# Initialize pygame(starting the game engine)
pygame.init()


# set the game display
GRID_SIZE = 25
CELL_SIZE = int(24 * 1.3) # this is one cell size ca be adjusted for visibility accordingly
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Navigation In Dynamic Envirement: Simulation")

# Color tuples
WHITE = (255, 255, 255)
GRAY = (20, 20, 20) # 0.1 is the alpha value for transparency
BLUE = (50, 100, 220)
ORANGE = (255, 165, 0)
PURPLE = (160, 32, 240)
BLACK = (0, 0, 0)

# Load and scale images to fit it in one cell
robot_img = pygame.image.load('robot.png')
robot_img = pygame.transform.smoothscale(robot_img, (CELL_SIZE, CELL_SIZE)) # smoothscale keeps the image quality high while resizing
home_img = pygame.image.load('home.png')
home_img = pygame.transform.smoothscale(home_img, (CELL_SIZE, CELL_SIZE))
bricks_img = pygame.image.load('bricks.png')
bricks_img = pygame.transform.smoothscale(bricks_img, (CELL_SIZE, CELL_SIZE))

# OBSTACLE AND GOAL
OBSTACLE = -1 #obstacle reward
GOAL = 100 #goal reward
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

# Action space
actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


# Q-learning parameters
alpha, gamma, epsilon = 0.3, 0.9, 0.5
'''
aplha: Learning rate- how much believe you keep in the current observation
gamma: Discount factor- how much you care about future rewards
epsilon: Exploration rate- how much you explore the environment
'''
episodes = 10000 # The more number of episodes, the more our agent explores the envirement (I must keep it bigger when envirement has more obstacles)

percentage_of_pause = 0.2 # during collision pause a robot for 50% and rest time let others move

# Let's initialize all Q values to 0 for all the cells
def init_Q():
    return {(i, j): {a: 0.0 for a in actions} for i in range(GRID_SIZE) for j in range(GRID_SIZE)}

# this function takes current position, action assigned, goal position and other goals positions as input and returns the 'next position' and 'reward' based on the action taken used to update Q-values.
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

        if episode % 10 == 0:
            screen.fill((255, 255, 255)) # let's fill the screen with white color
            draw_progress_bar(screen, episode, episodes, agent_name) #it will draw the updated progress bar everytime it is called as it also contains pygame.display.flip()

    return Q

# I am training 3 agents series wise, parallel training could be more efficient but for simplicity I am doing it series wise
print("Training agents...")
Q1 = train_agent((0, 0), goal_positions[0], goal_positions[1:], "Agent 1")
print("Trained Agent 1")
Q2 = train_agent((0, GRID_SIZE-1), goal_positions[1], [goal_positions[0], goal_positions[2]], "Agent 2")
print("Trained Agent 2")
Q3 = train_agent((GRID_SIZE-1, GRID_SIZE-1 ), goal_positions[2], goal_positions[:2], "Agent 3")
print("ALL Training Complete ^_^")

#Random moving obstacles:
human_img = pygame.image.load('human.png')
human_img = pygame.transform.smoothscale(human_img, (CELL_SIZE, CELL_SIZE))
# Initialize moving humans at random positions
number_of_humans= 70
human_positions = []
while len(human_positions) < number_of_humans:
    pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    if grid[pos] == 0 and pos not in goal_positions and pos not in [(0, 0), (0, GRID_SIZE - 1), (GRID_SIZE - 1, GRID_SIZE - 1)] and pos not in human_positions:
        human_positions.append(pos)

# this means, humans can avoid goals and static-obstacles:(they can collide with robots and other humans but robots can never collide with them that's what our algorithm does)
def move_human_randomly(pos):
    directions = list(actions.values())
    random.shuffle(directions) # It just suffles the order of actions => just like a card suffle  
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in goal_positions and grid[nx, ny] != OBSTACLE):
            return (nx, ny) # this is a valid move for the human
    return pos  # Stay in place if no valid move
#once called this is the new position of the human-'return pos'- later I will use it as laser detected value for agents


# Let's selsct a simple policy: agent will select only that action that will yield to max Q-value for that state(greedy policy)
policy1 = {s: max(Q1[s], key=Q1[s].get) for s in Q1} # dictionary of state to max valued action
policy2 = {s: max(Q2[s], key=Q2[s].get) for s in Q2}
policy3 = {s: max(Q3[s], key=Q3[s].get) for s in Q3}


# to draw the grid lines, fill grid color,and draw the obstacles and goals 
def draw_grid():
    screen.fill(BLACK) # grid background color
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRAY, rect, 1) # Grid lines
            if grid[i, j] == OBSTACLE:
                screen.blit(bricks_img, (j*CELL_SIZE, i*CELL_SIZE)) #static obstacles

    # Draw goals
    for goal, color in zip(goal_positions, goal_colors): #this is just like enumerate
        rect = pygame.Rect(goal[1]*CELL_SIZE, goal[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, rect)

# draw the robot(simply put it on it's current position)
def draw_entity(pos, image):
    x, y = pos[1]*CELL_SIZE, pos[0]*CELL_SIZE
    screen.blit(image, (x, y))

def draw_line_path(surface, path, color, width=4):
    points = [(pos[1]*CELL_SIZE + CELL_SIZE//2, pos[0]*CELL_SIZE + CELL_SIZE//2) for pos in path]
    pygame.draw.lines(surface, color, False, points, width)

# Define collision detection function
'''
we will use other_agent_positions(new) and moving_obstacle positions(new) and match them with our agent position if anywhere it matches then it's a collision because agent can never take action to off grid or to static obstacle or to othergoals or others starting positions.
'''

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
    

def main():
    clock = pygame.time.Clock()

    start_positions = [(0, 0), (0, GRID_SIZE-1), (GRID_SIZE-1,GRID_SIZE-1)]
    agent_positions = list(start_positions)
    paths = [[pos] for pos in agent_positions]
    policies = [policy1, policy2, policy3]
    goals = goal_positions

    path_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    path_colors = [(50, 100, 220, 180), (255, 165, 0, 180), (160, 32, 240, 180)]

    running = True

    while running:
        clock.tick(2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #Draw home, obstacles, grid lines.
        draw_grid()
        for start in start_positions:
            draw_entity(start, home_img)
            
        # Draw humans based on new move
        global human_positions
        human_positions = [move_human_randomly(pos) for pos in human_positions] #updated human positions after moving randomly
        for pos in human_positions:
           draw_entity(pos, human_img)
        # => here, new updated human positions are stored in list human_positions
                    
        
        path_surface.fill((0, 0, 0, 0))
        

        previous_positions = agent_positions.copy()  # Store previous positions for collision detection
        #this loop is for the robots to move one by one and after checking and getting updated they will be collected i.e their path for drawing
        for i in range(len(agent_positions)):
            if agent_positions[i] != goals[i]:
                action = policies[i][agent_positions[i]] # Get the action from the policy one by one as i changes for 1,2,3 robots
                
                other_goals = [g for idx, g in enumerate(goals) if idx != i]
                next_pos, _ = take_action(agent_positions[i], action, goals[i], other_goals)

                agent_positions[i] = next_pos # assigning i'th robot it's latest position after taking action according to i'th policy(corresponding policy)
        # so the upper loop upon running once(complete iterations), following the policies 1,2 and 3 it allows robots take one action each and update their new position
        
        
        # Detecting collisions-if detected, pause robot let the other robots move OR Just move it applied similar to => exploration-exploitation
        '''
         => new agent position is stored in a list of 3 => agent_positions #we will change it to safe place
         => new human position is stored in a list of 5 => human_positions #cann't be changed
        -we will change next move of agent one by one by utalizing new position of other agents and all humans
         => goal positions are stored in a list of 3 => goals
         => grid[nx, ny] == OBSTACLE => static wall, nx= x + dx, ny = y + dy
         => by checking the range of nx and ny we can check if it is out of bounds or not as:
            => if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                out of bounds
        '''
        
        
        # If collision will happen we will assign a safe position to the agents one by one by checking the 4 cells where it can move and out of which is safe to move randomly then following the policy , and we will do it one by one for each agent. If we cann't find a safe place to move then we will just pause the robot for a while and let others move.
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
                            paths[i].append(agent_positions[i]) #adding the new position to the path
                            found_safe_position = True
                            break
                
                    if not found_safe_position:
                        agent_positions[i] = previous_positions[i]  # it means no action taken
                        paths[i].append(agent_positions[i])
                        #print(f"No safe position found for Agent {i}, pausing")
            else:#good to go
                #no collision, so let's just append the current position to the path for drawing
                paths[i].append(agent_positions[i])
        
                        
        # Draw the paths and robots
        for i in range(len(agent_positions)):
            draw_line_path(path_surface, paths[i], path_colors[i])
            draw_entity(agent_positions[i], robot_img)

        screen.blit(path_surface, (0, 0)) # Draw paths on the main screen (0 px,0 px) represents from where to put the path_surface on the screen(here it is (0,0) means from the top left corner)
        pygame.display.flip() # like flipping a page going to next page(here tick(2) so 2 flips per second atmost)

        # Vallah we have reached then end the simulation (the main loop)
        if all(agent_positions[i] == goals[i] for i in range(len(agent_positions))):
            running = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
main()
