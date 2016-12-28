import os
import time
import json
import glob
import mmap
import logging
import numpy as np
from gym import error
logger = logging.getLogger(__name__)

__all__ = ['scan_recorded_traces', 'TraceRecordingReader']

class TraceRecordingReader:
    def __init__(self, directory):
        self.directory = directory
        self.binfiles = {}

    def close(self):
        for k in self.binfiles.keys():
            if self.binfiles[k] is not None:
                self.binfiles[k].close()
                self.binfiles[k] = None

    def get_binfile(self, fn):
        mm = self.binfiles.get(fn, None)
        if mm: return mm
        f = open(os.path.join(self.directory, fn), 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.binfiles[fn] = mm
        return mm

    def load_npy(self, o):
        mm = self.get_binfile(o['npyfile'])
        arr = np.ndarray.__new__(np.ndarray, o['shape'], dtype=o['dtype'], buffer=mm, offset=o['npyoff'], order='C')
        return arr

    def json_decode(self, o):
        o_type = o.get('__type', None)
        if o_type == 'ndarray':
            return self.load_npy(o)
        else:
            return o

    def get_recorded_batches(self):
        ret = []
        manifest_ptn = os.path.join(self.directory, 'openaigym.trace.*.manifest.json')
        trace_manifest_fns = glob.glob(manifest_ptn)
        logger.debug('Trace manifests %s %s', manifest_ptn, trace_manifest_fns)
        for trace_manifest_fn in trace_manifest_fns:
            trace_manifest_f = open(trace_manifest_fn, 'r')
            trace_manifest = json.load(trace_manifest_f)
            trace_manifest_f.close()
            ret += trace_manifest['batches']
        return ret

    def get_recorded_episodes(self, batch):
        batch_fn = os.path.join(self.directory, batch['fn'])
        batch_f = open(batch_fn, 'r')
        batch_d = json.load(batch_f, object_hook=self.json_decode)
        batch_f.close()
        return batch_d['episodes']

def scan_recorded_traces(directory, episode_cb=None, max_episodes=None):
    """
    Go through all the traces recorded to directory, and call episode_cb for every episode.
    Set max_episodes to end after a certain number (or you can just throw an exception from episode_cb
    if you want to end the iteration early)
    """
    rdr = TraceRecordingReader(directory)
    added_episode_count = 0
    for batch in rdr.get_recorded_batches():
        for ep in rdr.get_recorded_episodes(batch):
            episode_cb(ep['observations'], ep['actions'], ep['rewards'])
            added_episode_count += 1
            if max_episodes is not None and added_episode_count >= max_episodes: return
    rdr.close()
