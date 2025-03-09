import pygame
import matplotlib.pyplot as plt  
from Environment import Environment
from DQNAgent import DQNAgent

# Training Loop
def train_dqn(env, agent, episodes=1000, batch_size=32, fps=30):
    scores = []  # To store scores for each episode

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Process Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == env.obstacle_spawn_timer:
                    env.obstacles.add(env.Obstacle(env.floor_y, env.scale))

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Render the game
            env.render()

            # Control frame rate
            env.clock.tick(fps)

            if done:
                print(f"Episode: {episode + 1}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")
                scores.append(env.score)  # Store the score for this episode
                break

            agent.replay(batch_size)

        if (episode + 1) % 10 == 0:
            agent.save(f"dqn_model_episode_{episode + 1}.pth")

    # Plot the scores
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Progress")
    plt.show()


env = Environment()
state_size = len(env.get_state())

action_size = 2  # 0: Do nothing, 1: Jump
agent = DQNAgent(state_size, action_size)

# Train the agent
train_dqn(env, agent, episodes=25, fps=15)

# Close the environment
env.close()