from brian2 import *
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualise_connectivity(S):

    ### Lines going from source to target ###
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(18, 4))
    subplot(131)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    title('connections lines')


    ### Dot representing a connection ###
    subplot(132)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    title('connections')
    ### Size of dot representing the weight of the connection ###
    subplot(133)
    scatter(S.x_pre/um, S.x_post/um, S.w*0.5)
    xlabel('Source neuron position (um)')
    ylabel('Target neuron position (um)')
    title('connections weigths')

    plt.show()

def heatmap_1D(data1, data2):
    """
    Plots heatmaps of data1 and data2 with neuron indices on the y-axis.

    Parameters:
    - data1: np.ndarray of shape (neurons, time)
    - data2: np.ndarray of shape (neurons, time)
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot first heatmap
    im1 = axs[0].imshow(data1, aspect='auto', cmap='viridis', origin='lower',
                        vmin=0, vmax=1)
    axs[0].set_title('Output')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Neuron Index')
    plt.colorbar(im1, ax=axs[0], label=f'Voltage')

    # Plot second heatmap
    im2 = axs[1].imshow(data2, aspect='auto', cmap='plasma', origin='lower')
    axs[1].set_title('Input')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Neuron Index')
    plt.colorbar(im2, ax=axs[1], label='Input')

    plt.tight_layout()
    plt.show()

def plot_animate_1d(data1, theta,data2=None):
    """
    Animates the time evolution of activity data1 (u(x,t)) and optional inputs data2.
    
    Parameters:
    - data1: np.ndarray of shape (space,time)
    - data2: np.ndarray of shape (space,time)
    """

    # Spatial resolution and axis
    dx = 1  # Default spacing
    
    x_lim = data1.shape[0] * dx
    x = np.arange(0, x_lim, dx)

    # Set y-limits
    y_min = min(data1.min(), data2.min())
    y_max = max(data1.max(), data2.max())

    # Set up plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y_min, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("Activity/Input")

    data1 = data1.T
    data2 = data2.T

    line1, = ax.plot(x, data1[0], label="u(x)")
    line2, = ax.plot(x, data2[0], label="Input")

    ax.legend()
    # Plot the constant line (theta) if provided
    if theta is not None:
        ax.axhline(y=theta, color='r', linestyle='--', label=f"theta = {theta}")
        ax.legend()

    # Animation loop
    for i in range(data1.shape[0]):
        if i % 100 == 0:
            line1.set_ydata(data1[i])

            if data2 is not None:
                line2.set_ydata(data2[i])
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.002)

def save_activity_animation(data1, theta,  data2=None, filename="activity.gif"):
    """
    Animates and saves the time evolution of activity data1 (u(x,t)) and optional inputs data2.

    Parameters:
    - data1: np.ndarray of shape (space, time)
    - data2: np.ndarray of shape (space, time), optional
    - theta: float, optional threshold line
    - dx: spatial resolution
    - filename: name of the video file to save (e.g., .mp4 or .gif)
    """
    dx=1
    # Transpose to shape (time, space)
    data1 = data1.T
    if data2 is not None:
        data2 = data2.T

    # Spatial axis
    x = np.arange(0, data1.shape[1] * dx, dx)

    # y-limits
    y_min = min(data1.min(), data2.min()) if data2 is not None else data1.min()
    y_max = max(data1.max(), data2.max()) if data2 is not None else data1.max()

    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y_min, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("Activity/Input")

    line1, = ax.plot(x, data1[0], label="u(x)")
    line2 = None
    if data2 is not None:
        line2, = ax.plot(x, data2[0], label="Input")

    if theta is not None:
        ax.axhline(y=theta, color='r', linestyle='--', label=f"theta = {theta}")

    ax.legend()

    # Animation update function
    def update(i):
        line1.set_ydata(data1[i])
        if data2 is not None and line2 is not None:
            line2.set_ydata(data2[i])
        return [line1, line2] if line2 else [line1]

    # Create animation (skip frames for speed if needed)
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(0, data1.shape[0], 10),  # Every 5th frame
        interval=20,  # milliseconds
        blit=False
    )

    # Save animation using ffmpeg or pillow
    if filename.endswith(".mp4"):
        ani.save(filename, writer='ffmpeg', fps=30)
    elif filename.endswith(".gif"):
        ani.save("gaussian_N200.gif", writer='pillow', fps=30)
    else:
        raise ValueError("Filename must end with .mp4 or .gif")

    plt.close(fig)  # Close the figure after saving

def heatmap_1D_adj(data1, data2, G, time, fixed_timestep=0):
    """
    Plots a heatmap of data1 and line plots of data1 and data2 across neuron locations at a fixed timestep.

    Parameters:
    - data1: np.ndarray of shape (neurons, time)
    - data2: np.ndarray of shape (neurons, time)
    - G: object or namespace with attribute x (neuron locations), shape (neurons,)
    - time: np.ndarray of shape (time,), actual time values
    - fixed_timestep: int, index into time array
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    extent = [time[0], time[-1], G[0], G[-1]]

    # Heatmap of data1 with actual time on x-axis
    im1 = axs[0].imshow(data1, aspect='auto', cmap='viridis', origin='lower',
                        extent=extent,
                        vmin=0, vmax=1)
    axs[0].set_title('Membrane potential')
    axs[0].set_xlabel('Time t [s]')
    axs[0].set_ylabel('Space x [m]')

    plt.colorbar(im1, ax=axs[0])

    # Plot data1 at fixed timestep vs neuron location
    axs[1].plot(G, data1[:, fixed_timestep], label='Output', color='blue')
    axs[1].set_title(f'Snapshot at t = {time[500]:.2f} [s]')
    axs[1].set_xlabel('Space x [m]')
    axs[1].set_ylabel('Voltage [V]')
    axs[1].grid(True)

    axs[1].legend()

    # Plot data2 at fixed timestep vs neuron location
    axs[2].plot(G, data1[:, -1], label='Input', color='blue')
    axs[2].set_title(f'Snapshot at t = {time[-1]:.2f} [s]')
    axs[2].set_xlabel('Space x [m]')
    axs[2].set_ylabel('Voltage [V]')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def heatmap_1D_adj_2(data1, data2, G, time, fixed_timestep=0):
    """
    Plots a heatmap of data1 and line plots of data1 and data2 across neuron locations at a fixed timestep.

    Parameters:
    - data1: np.ndarray of shape (neurons, time)
    - data2: np.ndarray of shape (neurons, time)
    - G: array of neuron locations, shape (neurons,)
    - time: np.ndarray of shape (time,), actual time values
    - fixed_timestep: int, index into time array
    """

    # Set MATLAB-like font styles
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',  # 'Arial' if available
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11
    })

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    extent = [time[0], time[-1], G[0], G[-1]]

    # Heatmap of data1 with actual time on x-axis
    im1 = axs[0].imshow(data1, aspect='auto', cmap='viridis', origin='lower',
                        extent=extent, vmin=0, vmax=1)
    axs[0].set_title('Membrane Potential')
    axs[0].set_xlabel('Time t [s]')
    axs[0].set_ylabel('Space x [m]')
    plt.colorbar(im1, ax=axs[0])
    t = 4500
    # Plot data1 at fixed timestep vs neuron location
    axs[1].plot(G, data1[:, 0], color='blue')
    axs[1].set_title(f'Snapshot at t = {time[0]:.2f} [s]')
    axs[1].set_xlabel('Space x [m]')
    axs[1].set_ylabel('Voltage [V]')
    axs[1].grid(True)

    # Plot data2 at last timestep vs neuron location
    axs[2].plot(G, data1[:, t-1000], color='blue')
    axs[2].set_title(f'Snapshot at t = {time[t-1000]:.3f} [s]')
    axs[2].set_xlabel('Space x [m]')
    axs[2].set_ylabel('Voltage [V]')
    axs[2].grid(True)

    plt.tight_layout()
     # Save the figure
    plt.savefig('membrane_potential.png', dpi=300)  # You can change dpi or format if needed
    
    plt.show()

def plot_slider_1d(data1, data2,data3=None):
    """
    Creates an interactive slider plot of activity (data1) and optional inputs (data2),
    assuming shape (space, time) for both.
    """
    print(np.shape(data1), np.shape(data2), np.shape(data3))
    # Transpose to [time, space]
    data1 = data1.T
    data2 = data2.T

    if data3 is not None:
        data3 = data3.T
        
    dx = 1  # Assume uniform spacing
    x_lim = data1.shape[1] * dx
    x = np.arange(0, x_lim, dx)

    # Set y-axis limits
   
    y_min = min(data1.min(), data2.min())
    y_max = max(data1.max(), data2.max())

    

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=0.25)

    line1, = ax.plot(x, data1[0], label='u(x)')
    
    line2, = ax.plot(x, data2[0], label='Input(x)', linestyle='dashed')

    if data3 is not None:
        line3, = ax.plot(x, data3[0], label='h(x)')

    ax.legend()
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')

    # Slider setup
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, '', 0, data1.shape[0] - 1, valinit=0, valstep=1)
    slider.valtext.set_visible(False)

    # Reset button
    ax_reset = plt.axes([0.8, 0.02, 0.1, 0.04])
    reset_button = Button(ax_reset, 'Reset')

    # Time label
    time_label = plt.text(0.5, 0.05, f'Time Step: {slider.val/10} [ms]', transform=fig.transFigure, ha='center')

    def update(val):
        i = int(slider.val)
        line1.set_ydata(data1[i])
        line2.set_ydata(data2[i])
        if data3 is not None:
            line3.set_ydata(data3[i])
        time_label.set_text(f'Time Step: {i/10} [ms]')
        fig.canvas.draw_idle()

    def reset(event):
        slider.set_val(0)

    slider.on_changed(update)
    reset_button.on_clicked(reset)

    plt.show()