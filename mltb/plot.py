"""Plot tools."""
import matplotlib.pyplot as plt
import seaborn as sns

# see https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.twinx.html
def twin_axes_plot(values_1, label_1, values_2, label_2, shift_1=0, shift_2=0, 
                   title=None, label_x = 'x', color_1 = 'tab:red', color_2 = 'tab:blue'):
    """Create twin axes plot."""
    fig, ax1 = plt.subplots()
    
    if title != None:
        plt.title(title)

    ax1.set_xlabel(label_x)

    ax1.set_ylabel(label_1, color=color_1)
    ax1.plot(range(shift_1, len(values_1) + shift_1), values_1, color=color_1)
    ax1.tick_params(axis='y', labelcolor=color_1)

    ax2 = ax1.twinx()
    
    ax2.set_ylabel(label_2, color=color_2)
    ax2.plot(range(shift_2, len(values_2) + shift_2), values_2, color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    # otherwise the labels might be slightly clipped
    # see https://matplotlib.org/users/tight_layout_guide.html
    fig.tight_layout() 

    plt.show()

def boxplot(values, labels=None, title=None, xlabel=None, ylabel=None):
    """Create boxplot."""
   
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=values)

    if labels != None:
        plt.xticks(range(len(labels)), labels)

    if xlabel != None:
        ax.set(xlabel=xlabel)

    if ylabel != None:
        ax.set(ylabel=ylabel)

    if title !=  None:
        ax.set(title=title)

    plt.show()

def boxplot_dict(values_dict, title=None, xlabel=None, ylabel=None):
    """Create boxplot form dictionary."""
   
    values = []
    labels = []

    for key, value in values_dict.items():
        values.append(value)
        labels.append(key)
    
    boxplot(values, labels=labels, title=title, xlabel=xlabel, ylabel=ylabel)