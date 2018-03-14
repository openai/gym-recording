from gym_recording import playback
import matplotlib.pyplot as plt
import numpy as np
import glob, os

def handle_ep(observations, actions, rewards, infos):

    plt.ion()
    fig = plt.figure()
    fig.gca().set_aspect('equal', adjustable='box')
    ax = fig.gca() 

    xs = np.array([]) 
    ys = np.array([])
    alts = np.array([]) 

    # plot empty line to generate line object 
    line, = ax.plot(xs, ys) 

    plt.ioff() # turn off interactive mode 

    print('\n\n\n\nAn episode begins!')
    for obs, a, r, i in zip(observations, actions, rewards, infos):
        print('Obs: {} a: {} r: {} info: {}'.format(obs, a, r, i))
        if i:
            x = i['self_state']['lon']
            y = i['self_state']['lat']
            alt = i['relative_alt']
            xs = np.append(xs,x)
            ys = np.append(ys,y)
            alts = np.append(alts, alt)

    plt.plot(xs, ys, 'o-')


if __name__ == '__main__':
    path = '/tmp/gym/traces/'
    files = glob.glob(os.path.join(path, '*'))
    files.sort()
    
    playback.scan_recorded_traces(files[-1], handle_ep)
    plt.show()
