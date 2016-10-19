import os, logging, time
import gym
from gym_recording.wrappers import TraceRecordingWrapper
from gym_recording.playback import scan_recorded_traces
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_trace_recording():

    env = gym.make('CartPole-v0')
    env = TraceRecordingWrapper(env)
    recdir = env.directory
    agent = lambda ob: env.action_space.sample()

    for epi in range(10):
        ob = env.reset()
        for _ in range(100):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done: break
    env.close()

    counts = [0, 0]
    def handle_ep(observations, actions, rewards):
        counts[0] += 1
        counts[1] += observations.shape[0]
        logger.debug('Observations.shape={}, actions.shape={}, rewards.shape={}', observations.shape, actions.shape, rewards.shape)

    scan_recorded_traces(recdir, handle_ep)
    assert counts[0] == 10
    assert counts[1] > 100
