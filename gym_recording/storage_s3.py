import math, random, time, logging, re, base64, argparse, collections, sys, os, threading
import numpy as np
import boto3
import gym
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def upload_recording(directory, bucket):
    s3 = boto3.resource('s3')
    token = random_token(3)
    s3dir = os.path.join('recording', env_name, time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + token)
    logger.info('Scanning %s', recdir)
    b = s3.Bucket(bucket)
    for fn in os.listdir(recdir):
        logger.info('Upload %s', fn)
        b.upload_file(Filename=os.path.join(recdir, fn), Key=os.path.join(s3dir, fn))
    return 's3://' + bucket + '/' + s3dir


def download_recording(s3url):
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
