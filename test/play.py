from gym_recording import playback
import matplotlib.pyplot as plt
import numpy as np

def handle_ep(observations, actions, rewards, infos):

    plt.ion()
    fig = plt.figure()
    fig.gca().set_aspect('equal', adjustable='box')
    ax = fig.gca() 
    xs = np.array([]) 
    ys = np.array([]) 

    # plot empty line to generate line object 
    line, = ax.plot(xs,ys) 


    plt.ioff() # turn off interactive mode 

    print('\n\n\n\nAn episode begins!')
    for obs, a, r, i in zip(observations, actions, rewards, infos):
        print(': {} {} {} {}'.format(obs, a, r, i))
        x = i['self_state']['lon']
        y = i['self_state']['lat']
        xs = np.append(xs,x); 
        ys = np.append(ys,y); 
    plt.plot(xs, ys, '-o')


if __name__ == '__main__':
   # playback.scan_recorded_traces('./t', handle_ep)
    playback.scan_recorded_traces('/home/daniel/code/cogle/ddpg-craft/scripts/traces', handle_ep)
    plt.show()