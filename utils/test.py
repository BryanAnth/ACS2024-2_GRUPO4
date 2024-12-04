from tqdm import tqdm
import sys

MAX_STEPS = 5000



def test_pendulum(agent, pendulum, episodes=1000):
    for episode in tqdm(range(1, episodes + 1), desc="Testing"):
        state = pendulum.reset()
        done = False
        steps = 0

        #Testear mientras el cartpole este estabable y no se supere la unidad tiempo MAX_STEPS
        while steps < MAX_STEPS and not done:

            action = agent.choose_action(state, explore=False)
            next_state, reward, done, _ = pendulum.step(action)
            steps += 1
            state = next_state

        sys.stdout.write(f"\rEpisode {episode}: Balanced for {steps} steps")
        sys.stdout.flush()
    print()