from tqdm import tqdm
import sys

MAX_STEPS = 50



def test_pendulum(agent, pendulum, episodes=1000):
    for episode in tqdm(range(1, episodes + 1), desc="Testing"):
        state = pendulum.reset()
        done = False
        steps = 0

        while steps < MAX_STEPS:#not done:

            action = agent.choose_action(state, explore=False)
            next_state, reward, done, _ = pendulum.step(action)
            steps += 1

        sys.stdout.write(f"\rEpisode {episode}: Balanced for {steps} steps")
        sys.stdout.flush()
    print()