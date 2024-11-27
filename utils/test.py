from tqdm import tqdm

def test_pendulum(agent, pendulum, episodes=10):
    for episode in tqdm(range(1, episodes + 1)):
        state = pendulum.reset()
        done = False
        steps = 0

        while not done:

            action = agent.choose_action(state, explore=False)
            next_state, reward, done, _ = pendulum.step(action)
            steps += 1

        print(f"Episode {episode + 1}: Balanced for {steps} steps")