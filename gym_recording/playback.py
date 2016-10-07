import os
import time
import json
import glob
import logging
import numpy as np
from gym import error
logger = logging.getLogger(__name__)

__all__ = ['scan_recorded_traces']

class NpyReader:
    def __init__(self, directory):
        self.directory = directory
        self.bin_f_cache = None
        self.bin_fn_cache = None

    def close(self):
        self.bin_fn_cache = None
        if self.bin_f_cache is not None:
            self.bin_f_cache.close()
            self.bin_f_cache = None

    def open(self, fn):
        self.bin_fn_cache = fn
        self.bin_f_cache = open(os.path.join(self.directory, self.bin_fn_cache), 'rb')

    def json_decode(self, o):
        o_type = o.get('__type', None)
        if o_type == 'ndarray':
            o_npyfile = o.get('npyfile', None)
            o_npyoff = o.get('npyoff', None)
            if o_npyfile is not None and o_npyoff is not None:
                if self.bin_fn_cache != o_npyfile:
                    self.close()
                    self.open(o_npyfile)
                self.bin_f_cache.seek(o_npyoff)
                arr = np.load(self.bin_f_cache)
                return arr
            else:
                raise Exception('Unknown ndarray format {}'.format(o))
        else:
            return o


def scan_recorded_traces(directory, episode_cb=None, max_episodes=None):
    """
    Go through all the traces recorded to directory, and call episode_cb for every episode.
    Set max_episodes to end after a certain number (or you can just throw an exception from episode_cb
    if you want to end the iteration early)
    """
    rdr = NpyReader(directory)
    added_episode_count = 0
    manifest_ptn = os.path.join(directory, 'openaigym.trace.*.manifest.json')
    trace_manifest_fns = glob.glob(manifest_ptn)
    logger.debug('Trace manifests %s %s', manifest_ptn, trace_manifest_fns)
    for trace_manifest_fn in trace_manifest_fns:
        trace_manifest_f = open(trace_manifest_fn, 'r')
        trace_manifest = json.load(trace_manifest_f)
        trace_manifest_f.close()
        for batch in trace_manifest['batches']:
            batch_fn = os.path.join(directory, batch['fn'])
            batch_f = open(batch_fn, 'r')
            batch_d = json.load(batch_f, object_hook=rdr.json_decode)
            batch_f.close()
            for ep in batch_d['episodes']:
                episode_cb(ep['observations'], ep['actions'], ep['rewards'])
                added_episode_count += 1
                if max_episodes is not None and added_episode_count >= max_episodes: return

    rdr.close()
