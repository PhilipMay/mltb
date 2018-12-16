"""A collection of plot tools."""
import matplotlib.pyplot as plt

# see https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.twinx.html
def twin_axes_timeseries_plot(values_1, label_1, values_2, label_2, 
                              start_timestep_number=0, shift_1=0, shift_2=0, 
                              title=None, label_x = 'Step', color_1 = 'tab:red', color_2 = 'tab:blue'):
    """Create twin axes timeseries plot.
    
    Plots two different timeseries curves in one diagram but two different y-axes.

    Parameters
    ----------
    values_1 : array_like
        Values for the first timeseries curve.
    label_1 : str
        Label for the first timeseries curve.
    values_2 : array_like
        Values for the second timeseries curve.
    label_2 : str
        Label for the second timeseries curve.
    start_timestep_number : int, optional
        Number for first point in time. Default is 0.
    shift_1 : int, optional
        Number of timesteps to shift the first timeseries curve.
        Can be positive or negative. Default is 0.
    shift_2 : int, optional
        Number of timesteps to shift the second timeseries curve.
        Can be positive or negative. Default is 0.
    title : str, optional
        Title of the plot.
    label_x : str, optional
        Label for the x-axis (timeseries axis). Default is 'Step'.
    color_1 : str, optional
        Color of first timeseries curve. Default is 'tab:red'.
    color_2 : str, optional
        Color of second timeseries curve. Default is 'tab:blue'.
    """
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

def boxplot(values, labels=None, title=None, xlabel=None, ylabel=None, vert=True):
    """Create boxplot.
    
    Prints one or more boxplots in a single diagram.

    Parameters
    ----------
    values : array_like of numbers for one boxplot or array_like of array_like of numbers for several
        The values to draw the boxplot for. If you want to draw 
        more then one boxplot you have to give an array_like  
        of array_like with numbers.
    labels : str or array_like of str, optional
        The labels of the boxplots.
    title : str, optional
        Title of the plot.    
    xlabel : str, optional
        Label name of the x-axis.
    ylabel : str, optional
        Label name of the y-axis.
    vert : bool, optional
        If True (default), makes the boxes vertical. If False, everything is drawn horizontally.
    """
    _, ax = plt.subplots()
    
    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set(xlabel=xlabel)

    if ylabel is not None:
        ax.set(ylabel=ylabel)

    ax.boxplot(values, labels=labels, vert=vert)

    plt.grid(b=True, axis='y', linestyle='--')

    plt.xticks(rotation=90)

    plt.show()

def boxplot_dict(values_dict, title=None, xlabel=None, ylabel=None, vert=True):
    """Create boxplot form dictionary.
    
    Parameters
    ----------
    values_dict : dict with str to array_like
        Dictionary with one entry per box plot. The key (str) 
        is the name of the boxplot, the value (array_like) 
        contains the values to plot.
    title : str, optional
        Title of the plot.    
    xlabel : str, optional
        Label name of the x-axis.
    ylabel : str, optional
        Label name of the y-axis.
    vert : bool, optional
        If True (default), makes the boxes vertical. If False, everything is drawn horizontally.
    """
   
    values = []
    labels = []

    for key, value in values_dict.items():
        values.append(value)
        labels.append(key)
    
    boxplot(values, labels=labels, title=title, xlabel=xlabel, ylabel=ylabel, vert=vert)