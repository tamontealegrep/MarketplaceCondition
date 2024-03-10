
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .functions import percentile_range_data

#---------------------------------------------------------------------------------------------------

def create_subplots(plot_function, data, classes, num_rows, num_cols,
                    title="", title_font_size=16, fig_size=(5,5),
                    share_x = False, share_y = False,**kwargs):
    """
    Create a figure with a grid of subplots and apply a plotting function to each subplot.

    Parameters:
        plot_function (function): The function to apply to each subplot. It should accept two parameters:
                                  the subplot axis and the DataFrame for plotting.
        data (pandas.DataFrame): The DataFrame containing the data for plotting.
        classes (list): A list of class labels or identifiers.
        num_rows (int): The number of rows in the subplot grid.
        num_cols (int): The number of columns in the subplot grid.
        title (str, optional): The title for the entire figure.
        title_font_size (int, optional): The font size for the title of the entire figure.
        fig_size (tuple, optional): The size of the figure (width, height) in inches.
        **kwargs: Arbitrary keyword arguments to pass to plot_function.

    Returns:
        None

    Raises:
        ValueError: If the number of classes is greater than the number of subplots.
    """

    if len(classes) > num_rows * num_cols:
        raise ValueError("Number of classes are greater than the number of subplots.")
    
    # Special case when num_rows and num_cols are both 1
    if num_rows == 1 and num_cols == 1:
        fig, ax = plt.subplots(figsize=fig_size)
        plot_function(ax, data, classes[0], **kwargs)
        ax.set_title(title)
        plt.show()
        return

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=fig_size,sharex=share_x,sharey=share_y)
    if not (num_rows == 1 and num_cols == 1) and title != "":
        fig.suptitle(title, fontsize=title_font_size)  # Main title
    
    # Iterate over the subplots and apply the plot function
    for i in range(num_rows * num_cols):
        if i < len(classes):
            row = i // num_cols
            col = i % num_cols

            # Special case when num_rows or num_cols is 1
            if num_rows == 1:
                ax = axs[col]
            elif num_cols == 1:
                ax = axs[row]
            else:
                ax = axs[row, col]
            # Call the plot function
            plot_function(ax, data, classes[i],**kwargs)
        
        else:
            # Delete extra subplots
            fig.delaxes(axs.flat[i])
    
    fig.tight_layout()
    plt.show()

def binary_countplot_function(ax, data, class_label, **kwargs):
    """
    Plotting function to create a countplot using Seaborn for binary categorical variables (0 or 1).

    Parameters:
        ax (matplotlib.axes.Axes): The axes object of the subplot.
        data (pandas.Series or pandas.DataFrame): The data to create the countplot.
        class_label (str): The class label or identifier associated with the data.
        **kwargs: Additional keyword arguments to pass to sns.countplot().

    Returns:
        None
    """
    sns.countplot(x=data[class_label], data=data, ax=ax, **kwargs)
    ax.set_xticklabels(['False', 'True'])

def boxplot_function(ax, data, class_label,**kwargs):
    """
    Create a boxplot on the given subplot axis using the provided DataFrame.

    Parameters:
        ax (matplotlib.axes.Axes): The subplot axis to draw the boxplot.
        data (pandas.Series or pandas.DataFrame): The data for the boxplot.
        class_label (str): The label for the class corresponding to the data.
        **kwargs: Arbitrary keyword arguments to pass to seaborn.boxplot.

    Returns:
        None
    """
    sns.boxplot(data=data[class_label], ax=ax,**kwargs)
    ax.set_xlabel(class_label)

def histogram_function(ax, data, class_label,**kwargs):
    """
    Create a histogram on the given subplot axis using the provided DataFrame.

    Parameters:
        ax (matplotlib.axes.Axes): The subplot axis to draw the histogram.
        data (pandas.Series or pandas.DataFrame): The data for the histogram.
        class_label (str): The label for the class corresponding to the data.
        **kwargs: Arbitrary keyword arguments to pass to seaborn.histplot.

    Returns:
        None
    """
    sns.histplot(data=data[class_label], ax=ax,**kwargs)
    ax.set_xlabel(class_label)

def kdeplot_function(ax, data, class_label,**kwargs):
    """
    Create a KDE plot on the given subplot axis using the provided DataFrame.

    Parameters:
        ax (matplotlib.axes.Axes): The subplot axis to draw the KDE plot.
        data (pandas.Series or pandas.DataFrame): The data for the KDE plot.
        class_label (str): The label for the class corresponding to the data.
        **kwargs: Arbitrary keyword arguments to pass to seaborn.kdeplot.

    Returns:
        None
    """
    sns.kdeplot(data=data[class_label], ax=ax,**kwargs)
    ax.set_xlabel(class_label)

def boolean_correlation_heatmap_function(ax, data, column_1, column_2=None,**kwargs):
    """
    Create a heatmap subplot showing the correlation between two boolean columns.

    Parameters:
        ax (matplotlib.axes.Axes): The subplot axis to draw the heatmap.
        data (pandas.DataFrame): The DataFrame containing the data.
        column_1 (str): The name of the first boolean column.
        column_2 (str): The name of the second boolean column.

    Returns:
        None
    """
    # Extract column_2 from kwargs
    if column_2 is None:
        column_2 = kwargs.pop('column_2', None)
    if column_2 is None:
        raise ValueError("The 'column_2' keyword argument is missing.")
    df = data
    # Count the cases
    case_1_counts = df[(df[column_1] == False) & (df[column_2] == False)].shape[0]
    case_2_counts = df[(df[column_1] == False) & (df[column_2] == True)].shape[0]
    case_3_counts = df[(df[column_1] == True) & (df[column_2] == False)].shape[0]
    case_4_counts = df[(df[column_1] == True) & (df[column_2] == True)].shape[0]

    # Count the total occurrences of each column
    column_1_true_count = df[column_1].sum()
    column_1_false_count = df.shape[0] - column_1_true_count
    column_2_true_count = df[column_2].sum()
    column_2_false_count = df.shape[0] - column_2_true_count
    
    # Calculate the correlation percentages
    case_1_percentage = (case_1_counts / column_1_false_count) * (case_1_counts / column_2_false_count) * 100
    case_2_percentage = (case_2_counts / column_1_false_count) * (case_2_counts / column_2_true_count) * 100
    case_3_percentage = (case_3_counts / column_1_true_count) * (case_3_counts / column_2_false_count) * 100
    case_4_percentage = (case_4_counts / column_1_true_count) * (case_4_counts / column_2_true_count) * 100

    # Create the correlation matrix
    correlation_matrix = pd.DataFrame([[case_1_percentage, case_2_percentage],
                                       [case_3_percentage, case_4_percentage]],
                                      index=[False, True], columns=[False, True])

    # Create the heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', fmt='.1f', cbar=False, vmin=0, vmax=100, ax=ax,**kwargs)

    # Axes labels and title
    ax.set_xlabel(column_2)
    ax.set_ylabel(column_1)
    ax.set_title(f'{column_1} | {column_2}')

def plot_correlation_matrix(df, columns, fig_size=(10, 8), **kwargs):
    """
    Plot the correlation matrix for specified columns of a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list): A list of column names to include in the correlation matrix.
        fig_size (tuple, optional): The size of the figure (width, height) in inches. Default is (10, 8).
        **kwargs: Additional keyword arguments to pass to sns.heatmap().

    Returns:
        None
    """
    correlation_matrix = df[columns].corr()

    if 'cmap' in kwargs:
        cmap = kwargs.pop('cmap')
    else:
        cmap = 'coolwarm'

    plt.figure(figsize=fig_size)
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, vmax=1, vmin=-1, **kwargs)
    plt.title('Correlation Matrix')
    plt.show()

def plot_percentile_range(df, column, range_min=0, range_max=50, mode='both', title=True, title_size=16,log_scale=False):
    """
    Create a plot showing the range of percentiles for a given DataFrame column.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        column (str): The column to ve evaluated.
        range_min (int, optional): The minimum range to vary the percentile.  (default: 0).
        range_max (int, optional): The maximum range to vary the percentile (default: 50).
        mode (str, optional): The mode of operation. Possible values: 'min', 'max', 'both' (default: 'both').
        title (bool, optional): (default: True)
        title_size (int, optional): (default: 16)
        log_scale (bool,optional): (default: False)
    Returns:
        None
    """
    if range_min < 0:
        raise ValueError("range_min must be greater than 0")
    if range_max > 50:
        raise ValueError("range_min must be lower than 50")
    
    data = {"i": [], "max": [], "min": [], "mean":[], "std":[]}
    for i in range(range_min, range_max):
        if mode == 'min':
            info = percentile_range_data(df[column],0 + i, 100)
        elif mode == 'max':
            info = percentile_range_data(df[column],0, 100 - i)
        elif mode == 'both':
            info = percentile_range_data(df[column],0 + i, 100 - i)
        data["i"].append(i)
        data["max"].append(info.max())
        data["min"].append(info.min())
        data["mean"].append(info.mean())
        data["std"].append(info.std())

    percentile_info = pd.DataFrame(data, index=data['i'])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    if title:
        fig.suptitle(f"{column} Percentile Range Variation", fontsize=title_size)

    axs[0].plot(percentile_info['max'], label='Max')
    axs[0].plot(percentile_info['min'], label='Min')
    axs[0].plot(percentile_info['mean'], label='Mean')
    axs[0].set_title('Max | Mean | Min')
    axs[0].legend()
    if log_scale:
        axs[0].set_yscale('log')

    axs[1].plot(percentile_info['std'], label='Std', color='green')
    axs[1].set_title('Std')
    if log_scale:
        axs[1].set_yscale('log')

    plt.tight_layout()
    plt.show()
#---------------------------------------------------------------------------------------------------
