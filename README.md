# gym-recording



```Python
from gym_recording import playback, storage_s3

def handle_ep(observations, actions, rewards):
  #WRITEME

playback.scan_recorded_traces(storage_s3.download_recording('s3://...'), handle_ep)
```
