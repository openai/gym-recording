import gym
from gym_recording.wrappers import TraceRecordingWrapper

def main():
    env = gym.make('CartPole-v0')
    env = TraceRecordingWrapper(env, directory='./t', buffer_batch_size=10)
    print('log dir {}'.format(env.directory))
    print(env.__dict__)
    env.reset()
    for _ in range(10000):
        _, _, done, _ = env.step(env.action_space.sample()) # take a random action
        if done:
            env.reset()
    print('Done')

if __name__ == '__main__':
    main()
