"""Plot tools."""
import matplotlib.pyplot as plt

# see https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.twinx.html
def twin_axes_timeseries_plot(values_1, label_1, values_2, label_2, 
                              start_timestep_number=0, shift_1=0, shift_2=0, 
                              title=None, label_x = 'Step', color_1 = 'tab:red', color_2 = 'tab:blue'):
    """Create twin axes timeseries plot."""

    fig, ax1 = plt.subplots()
    
    if title != None:
        plt.title(title)

    ax1.set_xlabel(label_x)

    ax1.set_ylabel(label_1, color=color_1)
    ax1.plot(range(start_timestep_number + shift_1, len(values_1) + start_timestep_number + shift_1), 
             values_1, color=color_1)
    ax1.tick_params(axis='y', labelcolor=color_1)

    ax2 = ax1.twinx()
    
    ax2.set_ylabel(label_2, color=color_2)
    ax2.plot(range(start_timestep_number + shift_2, len(values_2) + start_timestep_number + shift_2), 
             values_2, color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    # otherwise the labels might be slightly clipped
    # see https://matplotlib.org/users/tight_layout_guide.html
    fig.tight_layout() 

    plt.show()

def boxplot(values, labels=None, title=None, xlabel=None, ylabel=None):
    """Create boxplot.
    
    Prints one or more boxplots in a single diagram.

    Parameters
    ----------
    values : iterable of numbers for one boxplot or iterable of iterable of numbers for several
        The values to draw the boxplot for. If you want to draw 
        more then one boxplot you have to give an iterable  
        of iterable with numbers.
    labels : str or iterable of str, optional
        The labels of the boxplots.
    xlabel : str, optional
        Label name of the x-axis.
    ylabel : str, optional
        Label name of the y-axis.
    """
   
    _, ax = plt.subplots()
    
    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set(xlabel=xlabel)

    if ylabel is not None:
        ax.set(ylabel=ylabel)

    ax.boxplot(values, labels=labels)

    plt.grid(b=True, axis='y', linestyle='--')

    plt.show()

def boxplot_dict(values_dict, title=None, xlabel=None, ylabel=None):
    """Create boxplot form dictionary.
    
    Parameters
    ----------
    values_dict : dictionary with one entry per box plot.
    """
   
    values = []
    labels = []

    for key, value in values_dict.items():
        values.append(value)
        labels.append(key)
    
    boxplot(values, labels=labels, title=title, xlabel=xlabel, ylabel=ylabel)