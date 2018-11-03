import matplotlib.pyplot

# see https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.twinx.html
def twin_axes_plot(val_1, label_1, val_2, label_2, label_x = 'x', color_1 = 'tab:red', color_2 = 'tab:blue'):
    """Create twin axes plot."""
    fig, ax1 = matplotlib.pyplot.subplots()
    
    ax1.set_xlabel(label_x)

    ax1.set_ylabel(label_1, color=color_1)
    ax1.plot(range(len(val_1)), val_1, color=color_1)
    ax1.tick_params(axis='y', labelcolor=color_1)

    ax2 = ax1.twinx()
    
    ax2.set_ylabel(label_2, color=color_2)
    ax2.plot(range(len(val_2)), val_2, color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    # otherwise the labels might be slightly clipped
    # see https://matplotlib.org/users/tight_layout_guide.html
    fig.tight_layout() 
    
    matplotlib.pyplot.show()