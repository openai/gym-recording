import math, random, time, logging, re, base64, argparse, collections, sys, os, threading
import numpy as np
import boto3
import gym
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ['upload_recording', 'download_recording']

cookie_rng = random.SystemRandom() # initialized from /dev/urandom
def random_token(l=3):
    return ''.join([cookie_rng.choice('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz') for i in range(l)])



def upload_recording(directory, env_name, bucket):
    """
    Upload the recording saved in directory to bucket. Returns an s3://... URL
    """
    s3 = boto3.resource('s3')
    token = random_token(3)
    s3dir = os.path.join('recording', env_name, time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + token)
    logger.info('Scanning %s', directory)
    b = s3.Bucket(bucket)
    for fn in os.listdir(directory):
        logger.info('Upload %s to %s', fn, os.path.join(s3dir, fn))
        b.upload_file(Filename=os.path.join(directory, fn), Key=os.path.join(s3dir, fn))
    return 's3://' + bucket + '/' + s3dir


def download_recording(s3url):
    """
    Download the recording saved in s3url to a directory in /tmp. It'll reuse the cached
    recording if it's already downloaded. Returns a directory
    """
    s3 = boto3.resource('s3')
    m = re.match(r's3://([\w\-]+)/([\w\-\.\/]+)', s3url)
    if not m:
        raise Exception('Invalid s3 URL')
    bucket = m.group(1)
    bucketdir = m.group(2)
    directory = os.path.join('/tmp', bucketdir)
    if not os.access(directory, os.R_OK):
        directory_tmp = directory + '.tmp{}'.format(os.getpid())
        os.makedirs(directory_tmp)
        b = s3.Bucket(bucket)
        for fn in b.objects.filter(Prefix=bucketdir+'/'):
            logger.info('Download %s to %s', fn.key, directory)
            localfn = os.path.join(directory_tmp, os.path.basename(fn.key))
            b.download_file(Key=fn.key, Filename=localfn)
        os.rename(directory_tmp, directory)
    return directory
