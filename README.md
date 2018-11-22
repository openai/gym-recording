**Status:** Archive (code is provided as-is, no updates expected)

# gym-recording

A Python package to capture the sequences of actions and observations on a [Gym](https://github.com/openai/gym) environment
by wrapping it in a `TraceRecordingWrapper`, like this:

```Python
import gym
from gym_recording.wrappers import TraceRecordingWrapper

def main():
    env = gym.make('CartPole-v0')
    env = TraceRecordingWrapper(env)
    # ... exercise the environment
```

It will save recorded traces in a directory, which it will print with `logging`.
You can get the directory name from your code as `env.directory`.

Later you can play back the recording with code like the following, which runs a callback for every episode.

```Python
from gym_recording import playback

def handle_ep(observations, actions, rewards):
  # ... learn a model

playback.scan_recorded_traces(directory, handle_ep)
```

You can also use the storage_s3 module to upload and download traces from S3, so you it can run across machines.

```Python
from gym_recording import playback, storage_s3

def main():
    env = gym.make('CartPole-v0')
    env = TraceRecordingWrapper(env)
    # ... exercise the environment

    s3url = storage_s3.upload_recording(env.directory, env.spec.id, 'openai-example-traces')
    # ... store s3url in a database

    # ... Switch to another machine

    # ... load s3url from a database

    def handle_ep(observations, actions, rewards):
      # ... learn a model
    playback.scan_recorded_traces(storage_s3.download_recording(s3url), handle_ep)

```
