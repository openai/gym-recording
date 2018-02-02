from gym_recording import playback
import matplotlib.pyplot as plt
import numpy as np

def handle_ep(observations, actions, rewards, infos):

    plt.ion()
    fig = plt.figure(); 
    ax = fig.gca() 
    xs = np.array([]) 
    ys = np.array([]) 

    # plot empty line to generate line object 
    line, = ax.plot(xs,ys) 


    plt.ioff() # turn off interactive mode 

    print('\n\n\n\nAn episode begins!')
    for obs, a, r, i in zip(observations, actions, rewards, infos):
        print(': {} {} {} {}'.format(obs, a, r, i))
        x = i['self_state']['lat']
        y = i['self_state']['lon']
        xs = np.append(xs,x); 
        ys = np.append(ys,y); 
    plt.plot(xs, ys, 'r')


if __name__ == '__main__':
   playback.scan_recorded_traces('./t', handle_ep)
    plt.show()