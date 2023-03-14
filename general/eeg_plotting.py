"""
eeg_plotting.py

Author:
    Daniel Schonhaut
    
Dependencies: 
    Python 3.6, numpy, matplotlib

Description: 
    Functions for plotting EEG data stored as numpy ndarrays.

Last Edited: 
    10/28/18
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_trace(dat, 
               spike_dat=None, 
               start=0, 
               duration=None, 
               stop=None, 
               nwin=6, 
               x_units='secs', 
               x_inds=None, 
               x_lab=None, 
               y_lab=None, 
               sampling_rate=2000, 
               step=1,
               colors=None, 
               spike_colors=None, 
               alphas=None, 
               linewidths=None, 
               linestyles=None, 
               markers=None, 
               markersizes=None, 
               markeredgewidths=None, 
               legend_labels=None, 
               legend_size=8, 
               dpi=300, 
               figsize=[12, 16], 
               title=None, 
               same_yaxis=True, 
               sub_mean=True,
               create_spread=True,
               conv_to_muv=True,
               tickfontsize=10, 
               labelfontsize=12, 
               figfontsize=14,
               fig=None, 
               ax=None):
    """Plot one or more raw traces over time, splitting time into a specified
    number of subplots.
    
    Parameters
    ----------
    dat : numpy.ndarray
        An m x n array of traces to plot, where m is the number of traces and n
        is the number of timepoints.
    spike_dat : numpy.ndarray
        An m x n array of spike trains to plot, where m is the number of units
        and n is the number of timepoints.
    start : int
        The start time for each trace, in secs or samples depending on the value
        of x_units.
    duration : int
        The duration of each trace, in secs or samples depending on the value of
        x_units. If no duration or stop time are provided then the whole trace
        from the start time is plotted.
    stop : int
        The stop time for each trace, in secs or samples depending on the value
        of x_units. If stop and duration are both passed as inputs then the 
        duration is ignored.
    nwin : int
        The number of subplot rows.
    x_units : str
        Units to plot along the x-axis, should be 'samples' or 'ms' or 'secs' 
        or 'mins'.
    x_inds : numpy.ndarray or list
        Indices to use for the x-axis; should match the length of the second
        dimension of dat. If none are given, these are determined by x_units.
    x_lab : str
        x-axis label. If none is given, this is determined by x_units.
    y_lab : str
        y-axis label. If none is given, no axis label is shown.
    sampling_rate : int
        Sampling rate of the data in Hz.
    step : int
        Plot every step-th timepoint.
    colors : list
        A list of colors to use for the traces, in order; should match the
        length of the first dimension of dat. If no colors are given then the 
        default matplotlib colors are used.
    alphas : list
        A list of alpha values to use for the traces, in order; should match the
        length of the first dimension of dat. Default is 1.
    linewidths : list
        A list of line widths to use for the traces, in order; should match the
        length of the first dimension of dat. Default is 0.8.
    linestyles : list
        A list of line styles to use for the traces, in order; should match the
        length of the first dimension of dat. Default is a solid line.
    markers : list
        A list of markers to use for the units in spike_dat, in order; should 
        match the length of the first dimension of spike_dat. If no markers are 
        given then vertical lines are used by default.
    markersizes : list
        A list of marker sizes to use for the traces, in order; should match the 
        length of the first dimension of dat. Default is 4.
    markeredgewidths : list
        A list of marker edge widths to use for the traces, in order; should
        match the length of the first dimension of dat. Default is 0.5.
    legend_labels : list
        A list of labels to use for the legend, in order; should be <= the 
        length of the first dimension of dat. If no legend_labels are given then
        no legend is made.    
    legend_size : int or float    
        Legend size, only relevant if legend labels are provided.
    figsize : list or tuple
        (x, y) size of the figure in inches.
    title : str
        Axis title.
    same_yaxis : bool
        If True, all subplots use the same ymin and ymax. Otherwise ylims
        are set optimally for each subplot.
    sub_mean : bool
        If True, performs mean subtraction of each channel, over time.
    create_spread : bool
        If True, separates channels along the y-axis instead of plotting them
        one on top of the other.
    conv_to_muv : bool
        If True, multiplies all values by 1e6 to convert V to µV.
        
    Returns
    -------
    fig, ax
        matplotlib figure and axis handles for further plot manipulation.
    """
    if len(dat.shape) == 1:
        dat = np.expand_dims(dat, axis=0)
    n_traces = dat.shape[0]
    n_samples = dat.shape[1]
    if spike_dat is not None:
        if len(spike_dat.shape) == 1:
            spike_dat = np.expand_dims(spike_dat, axis=0)
        n_units = spike_dat.shape[0]
    else:
        n_units = 0
    
    # Convert units from volts to microvolts.
    if conv_to_muv:
        dat = dat * 1e6
    
    # Subtract the mean from each channel.
    if sub_mean:
        dat = dat - np.expand_dims(np.mean(dat, axis=1), axis=1)
        
    # Spread out each channel on the y-axis.
    if create_spread:
        offsets = np.arange(n_traces) * 1.2 * (np.percentile(dat, 99) - np.percentile(dat, 1))
        dat = (dat + np.expand_dims(offsets, axis=-1))
        
    # Determine the x-axis units.
    if x_units == 'samples':
        if stop:
            stop = min(stop, n_samples)
        else:
            if duration:
                stop = min(start + duration, n_samples)
            else:
                stop = n_samples
        start = start
        if x_inds is None:
            x_inds = np.arange(n_samples) + 1
        if not x_lab:
            x_lab = 'Time (samples at {} Hz)'.format(sampling_rate)
    elif x_units == 'ms':
        if stop:
            stop = min(stop * (sampling_rate * 1e-3), n_samples)
        else:
            if duration:
                stop = min((start + duration) * (sampling_rate * 1e-3), n_samples)
            else:
                stop = n_samples
        start = start * (sampling_rate * 1e-3)
        if x_inds is None:
            x_inds = (np.arange(n_samples) + 1) / (sampling_rate * 1e-3)
        if not x_lab:
            x_lab = 'Time (ms)'
    elif x_units == 'secs':
        if stop:
            stop = min(stop * sampling_rate, n_samples)
        else:
            if duration:
                stop = min((start + duration) * sampling_rate, n_samples)
            else:
                stop = n_samples
        start = start * sampling_rate
        if x_inds is None:
            x_inds = (np.arange(n_samples) + 1) / sampling_rate
        if not x_lab:
            x_lab = 'Time (secs)'
    elif x_units == 'mins':
        if stop:
            stop = min(stop * (sampling_rate * 60), n_samples)
        else:
            if duration:
                stop = min((start + duration) * (sampling_rate * 60), n_samples)
            else:
                stop = n_samples
        start = start * (sampling_rate * 60)
        if x_inds is None:
            x_inds = (np.arange(n_samples) + 1) / (sampling_rate * 60)
        if not x_lab:
            x_lab = 'Time (mins)'
    start = int(start)
    stop = int(stop)
    
    # Split duration into nwin intervals.
    wins = np.linspace(start, stop, nwin+1, dtype=int)
    
    # Determine the y-axis limits.
    if same_yaxis:
        dat_min = np.min(dat[:, start:stop])
        dat_max = np.max(dat[:, start:stop])
        ymin = dat_min - (0.1 * abs(dat_min))
        ymax = dat_max + (0.1 * abs(dat_max))
        if ymin == 0:
            ymin = -0.1 * ymax
        if ymax == 0:
            ymax = 0.1 * -ymin
    else:
        ymin = []
        ymax = []
        for x in range(nwin):
            dat_min = np.nanmin(dat[:, wins[x]:wins[x+1]])
            dat_max = np.nanmax(dat[:, wins[x]:wins[x+1]])
            ymin_ = dat_min - (0.1 * abs(dat_min))
            ymax_ = dat_max + (0.1 * abs(dat_max))
            if ymin_ == 0:
                ymin_ = -0.1 * ymax
            if ymax_ == 0:
                ymax_ = 0.1 * -ymin
            ymin.append(ymin_)
            ymax.append(ymax_)
    
    # Create default lists for plot aesthetics that weren't passed as inputs.
    if not alphas:
        alphas = [1 for i in range(n_traces)]
    if not linewidths:
        linewidths = [0.8 for i in range(n_traces)]
    if not linestyles:
        linestyles = ['-' for i in range(n_traces)]
    if not markers: 
        markers = ['|' for i in range(n_units)]
    if not markersizes:
        markersizes = [12 for i in range(n_units)]
    if not markeredgewidths:
        markeredgewidths = [0.8 for i in range(n_units)]
    for i in range(n_units):
        if same_yaxis:
            set_val = ymax - ((ymax - ymin) * 0.15)
        else:
            x = ymax.index(min(ymax))
            set_val = ymax[x] - ((ymax[x] - ymin[x]) * 0.15)
        spike_dat[i, spike_dat[i, :]>0] = set_val
        spike_dat[i, spike_dat[i, :]==0] = np.nan
    if legend_labels and not legend_size:
        legend_size = 8
        
    # Create the plot.
    if (fig is None) and (ax is None):
        plt.close()
        fig, ax = plt.subplots(nwin, 1, dpi=dpi, figsize=figsize)  
        ax = np.ravel(ax)

    for x in range(len(ax)):
        start = wins[x]
        stop = wins[x+1]
        for i in range(n_traces):
            if colors:
                ax[x].plot(x_inds[start:stop][::step], 
                           dat[i, start:stop][::step], 
                           linewidth=linewidths[i], 
                           linestyle=linestyles[i], 
                           color=colors[i], 
                           alpha=alphas[i])
            else:
                ax[x].plot(x_inds[start:stop][::step], 
                           dat[i, start:stop][::step], 
                           linewidth=linewidths[i], 
                           linestyle=linestyles[i],
                           alpha=alphas[i])
        for i in range(n_units):
            if spike_colors:
                ax[x].plot(x_inds[start:stop][::step], 
                           spike_dat[i, start:stop][::step], 
                           linewidth=0, 
                           marker=markers[i], 
                           markerfacecolor=spike_colors[i], 
                           markeredgecolor=spike_colors[i],
                           alpha=alphas[i], 
                           markersize=markersizes[i], 
                           markeredgewidth=markeredgewidths[i])
            else:
                ax[x].plot(x_inds[start:stop][::step], 
                           spike_dat[i, start:stop][::step], 
                           linewidth=0, 
                           marker=markers[i], 
                           alpha=alphas[i], 
                           markersize=markersizes[i], 
                           markeredgewidth=markeredgewidths[i])
        
        if same_yaxis:        
            ax[x].axis([x_inds[start], x_inds[stop-1], ymin, ymax])
        else:
            ax[x].axis([x_inds[start], x_inds[stop-1], ymin[x], ymax[x]])
        ax[x].tick_params(labelsize=tickfontsize)
        ax[x].spines['right'].set_visible(False)
        ax[x].spines['top'].set_visible(False)
        if y_lab:
            ax[x].set_ylabel(y_lab, fontsize=labelfontsize, labelpad=8)
        else:
            ax[x].set_ylabel('μV', fontsize=labelfontsize, labelpad=8)
    
    ax[-1].set_xlabel(x_lab, fontsize=labelfontsize, labelpad=8)
    if legend_labels:
        ax[0].legend(legend_labels, loc=1, fontsize=tickfontsize, prop={'size': legend_size}, bbox_to_anchor=[1, 1.1])
    if title:
        fig.suptitle(title, fontsize=figfontsize)
    fig.tight_layout()
    
    return fig, ax


def _plot_trace(dat, start=0, duration=None, stop=None, x_units='secs', 
                x_inds=None, x_lab=None, y_lab=None, sampling_rate=2000, 
                colors=None, alphas=None, linewidths=None, linestyles=None, 
                markers=None, markersizes=None, markeredgewidths=None, 
                legend_labels=None, legend_size=8, dpi=300, figsize=[12, 4], 
                tickfontsize=10, labelfontsize=12, figfontsize=14, 
                sub_mean=True, create_spread=True, title=None, step=1, 
                fig=None, ax=None):
    """Plot one or more raw traces over time.
    
    Parameters
    ----------
    dat : numpy.ndarray
        An m x n array of traces to plot, where m is the number of traces and n
        is the samples for each trace.
    start : int
        The start time for each trace, in secs or samples depending on the value
        of x_units.
    duration : int
        The duration of each trace, in secs or samples depending on the value of
        x_units. If no duration or stop time are provided then the whole trace
        from the start time is plotted.
    stop : int
        The stop time for each trace, in secs or samples depending on the value
        of x_units. If stop and duration are both passed as inputs then the 
        duration is ignored.
    x_units : str
        Units to plot along the x-axis, should be 'samples' or 'ms' or 'secs' 
        or 'mins'.
    x_inds : numpy.ndarray or list
        Indices to use for the x-axis; should match the length of the second
        dimension of dat. If none are given, these are determined by x_units.
    x_lab : str
        x-axis label. If none is given, this is determined by x_units.
    y_lab : str
        y-axis label. If none is given, no axis label is shown.
    sampling_rate : int
        Sampling rate of the data in Hz. 
    colors : list
        A list of colors to use for the traces, in order; should match the
        length of the first dimension of dat. If no colors are given then the 
        default matplotlib colors are used.
    alphas : list
        A list of alpha values to use for the traces, in order; should match the
        length of the first dimension of dat. Default is 1.
    linewidths : list
        A list of line widths to use for the traces, in order; should match the
        length of the first dimension of dat. Default is 0.7.
    linestyles : list
        A list of line styles to use for the traces, in order; should match the
        length of the first dimension of dat. Default is a solid line.
    markers : list
        A list of markers to use for the traces, in order; should match the 
        length of the first dimension of dat. If no markers are given then only
        lines are plotted. If markers are wanted for some but not all traces,
        'None' should be used for the dat trace indices where lines are wanted.
        The '|' marker symbol is intended for plotting Boolean traces (e.g. 
        spike trains) and is implemented somewhat uniquely, with 0 values 
        converted to np.nan, 1 values plotted 85% of the way up the y-axis, and 
        different default values used for markeredgewidth and markersize. 
    markersizes : list
        A list of marker sizes to use for the traces, in order; should match the 
        length of the first dimension of dat. Default is 4.
    markeredgewidths : list
        A list of marker edge widths to use for the traces, in order; should
        match the length of the first dimension of dat. Default is 0.5.
    legend_labels : list
        A list of labels to use for the legend, in order; should be <= the 
        length of the first dimension of dat. If no legend_labels are given then
        no legend is made.    
    legend_size : int or float    
        Legend size, only relevant if legend labels are provided.
    figsize : list or tuple
        (x, y) size of the figure in inches.
    title : str
        Axis title.
        
    Returns
    -------
    fig, ax
        matplotlib figure and axis handles for further plot manipulation. 
        
    Examples
    --------
    freq = 4
    sampling_rate = 200
    duration = 2
    n_samples = sampling_rate * duration
    sin_wave = np.sin(np.linspace(-np.pi, np.pi, n_samples) * duration * freq)
    cos_wave = np.cos(np.linspace(-np.pi, np.pi, n_samples) * duration * freq)
    dat = np.vstack((sin_wave, cos_wave))
    
    # Plot the first 1 sec of data 
    fig, ax = plot_trace(dat, start=0, duration=1, x_units='secs', 
                         sampling_rate=sampling_rate, colors=['C0', 'C2'], 
                         markers=[None, '+'], 
                         legend_labels=['sin_wave', 'cos_wave'])                                         
    """
    n_traces = dat.shape[0]
    n_samples = dat.shape[1]
    
    # Convert units from volts to microvolts.
    dat = dat * 1e6
    
    # Subtract the mean from each channel.
    if sub_mean:
        dat = dat - np.expand_dims(np.mean(dat, axis=1), axis=1)
        
    # Spread out each channel on the y-axis.
    if create_spread:
        offsets = np.arange(n_traces) * 1.2 * (np.percentile(dat, 99) - np.percentile(dat, 1))
        dat = (dat + np.expand_dims(offsets, axis=-1))
        
    # Determine the x-axis units.
    if x_units == 'samples':
        if stop:
            stop = min(stop, n_samples)
        else:
            if duration:
                stop = min(start + duration, n_samples)
            else:
                stop = n_samples
        start = start
        if x_inds is None:
            x_inds = np.arange(n_samples) + 1
        if not x_lab:
            x_lab = 'Time (samples at {} Hz)'.format(sampling_rate)
    elif x_units == 'ms':
        if stop:
            stop = min(stop * (sampling_rate * 1e-3), n_samples)
        else:
            if duration:
                stop = min((start + duration) * (sampling_rate * 1e-3), n_samples)
            else:
                stop = n_samples
        start = start * (sampling_rate * 1e-3)
        if x_inds is None:
            x_inds = (np.arange(n_samples) + 1) / (sampling_rate * 1e-3)
        if not x_lab:
            x_lab = 'Time (ms)'
    elif x_units == 'secs':
        if stop:
            stop = min(stop * sampling_rate, n_samples)
        else:
            if duration:
                stop = min((start + duration) * sampling_rate, n_samples)
            else:
                stop = n_samples
        start = start * sampling_rate
        if x_inds is None:
            x_inds = (np.arange(n_samples) + 1) / sampling_rate
        if not x_lab:
            x_lab = 'Time (secs)'
    elif x_units == 'mins':
        if stop:
            stop = min(stop * (sampling_rate * 60), n_samples)
        else:
            if duration:
                stop = min((start + duration) * (sampling_rate * 60), n_samples)
            else:
                stop = n_samples
        start = start * (sampling_rate * 60)
        if x_inds is None:
            x_inds = (np.arange(n_samples) + 1) / (sampling_rate * 60)
        if not x_lab:
            x_lab = 'Time (mins)'
    start = int(start)
    stop = int(stop)
    
    # Determine the y-axis limits.
    if markers:
        keep = [i for i, val in enumerate(markers) if val != '|']
    else:
        keep = np.arange(dat.shape[0], dtype=int)
    dat_min = np.min(dat[keep, start:stop])
    dat_max = np.max(dat[keep, start:stop])
    ymin = dat_min - (0.1 * abs(dat_min))
    ymax = dat_max + (0.1 * abs(dat_max))
    if ymin == 0:
        ymin = -0.1 * ymax
    if ymax == 0:
        ymax = 0.1 * -ymin
    
    # Create default lists for plot aesthetics that weren't passed as inputs.
    if not colors:
        colors = [None for i in range(n_traces)]
    if not alphas:
        alphas = [1 for i in range(n_traces)]
    if not linewidths:
        linewidths = [0.7 for i in range(n_traces)]
    if not linestyles:
        linestyles = ['-' for i in range(n_traces)]
    if not markers: 
        markers = [None for i in range(n_traces)]
    else:
        for i, marker in enumerate(markers):
            # For plotting a Boolean trace
            if marker == '|':
                dat[i, dat[i, :]==0] = np.nan
                dat[i, dat[i, :]==1] = ymax - ((ymax - ymin) * 0.15)
    if not markersizes:
        markersizes = [20 if markers[i]=='|' else 4 
                       for i in range(n_traces)]
    if not markeredgewidths:
        markeredgewidths = [0.3 if markers[i]=='|' else 0.5 
                            for i in range(n_traces)]
    if legend_labels and not legend_size:
        legend_size = 8
        
    # Create the plot.
    if fig is None and ax is None:     
        fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize)  
    ax = np.ravel(ax) 
    traces = np.arange(n_traces)
    for i in traces:
        if markers[i]:
            if colors[i]:
                ax[0].plot(x_inds[start:stop][::step], dat[i, start:stop][::step], 
                           linewidth=0, marker=markers[i], 
                           markerfacecolor=colors[i], markeredgecolor=colors[i],
                           alpha=alphas[i], markersize=markersizes[i], 
                           markeredgewidth=markeredgewidths[i])
            else:
                ax[0].plot(x_inds[start:stop][::step], dat[i, start:stop][::step], 
                           linewidth=0, marker=markers[i], 
                           alpha=alphas[i], markersize=markersizes[i], 
                           markeredgewidth=markeredgewidths[i])
        else:
            if colors[i]:
                ax[0].plot(x_inds[start:stop][::step], dat[i, start:stop][::step], 
                           linewidth=linewidths[i], linestyle=linestyles[i], 
                           color=colors[i], alpha=alphas[i])
            else:
                ax[0].plot(x_inds[start:stop][::step], dat[i, start:stop][::step], 
                           linewidth=linewidths[i], linestyle=linestyles[i],
                           alpha=alphas[i])
    ax[0].axis([x_inds[start], x_inds[stop-1], ymin, ymax])
    ax[0].tick_params(labelsize=tickfontsize)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_xlabel(x_lab, fontsize=labelfontsize, labelpad=8)
    if y_lab:
        ax[0].set_ylabel(y_lab, fontsize=labelfontsize, labelpad=8)
    if legend_labels:
        ax[0].legend(legend_labels, loc=1, fontsize=tickfontsize, prop={'size': legend_size}, bbox_to_anchor=[1, 1.1])
    if title:
        ax[0].set_title(title, fontsize=figfontsize)
    
    return fig, ax
    
    
def plot_trace2(dat, spike_dat=None, start=0, duration=None, stop=None, nwin=6, x_units='secs', 
                x_inds=None, x_lab=None, y_lab=None, sampling_rate=2000, 
                colors=None, spike_colors=None, alphas=None, linewidths=None, linestyles=None, 
                markers=None, markersizes=None, markeredgewidths=None, 
                legend_labels=None, legend_size=8, dpi=300, figsize=[12, 16], 
                title=None, same_yaxis=True, sub_mean=True, create_spread=True,
                tickfontsize=10, labelfontsize=12, figfontsize=14, 
                step=1, fig=None, ax=None):
    """Plot one or more raw traces over time, splitting time into a specified
    number of subplots.
    
    Parameters
    ----------
    dat : numpy.ndarray
        An m x n array of traces to plot, where m is the number of traces and n
        is the samples for each trace.
    start : int
        The start time for each trace, in secs or samples depending on the value
        of x_units.
    duration : int
        The duration of each trace, in secs or samples depending on the value of
        x_units. If no duration or stop time are provided then the whole trace
        from the start time is plotted.
    stop : int
        The stop time for each trace, in secs or samples depending on the value
        of x_units. If stop and duration are both passed as inputs then the 
        duration is ignored.
    nwin : int
        The number of subplots to make.
    x_units : str
        Units to plot along the x-axis, should be 'samples' or 'ms' or 'secs' 
        or 'mins'.
    x_inds : numpy.ndarray or list
        Indices to use for the x-axis; should match the length of the second
        dimension of dat. If none are given, these are determined by x_units.
    x_lab : str
        x-axis label. If none is given, this is determined by x_units.
    y_lab : str
        y-axis label. If none is given, no axis label is shown.
    sampling_rate : int
        Sampling rate of the data in Hz. 
    colors : list
        A list of colors to use for the traces, in order; should match the
        length of the first dimension of dat. If no colors are given then the 
        default matplotlib colors are used.
    alphas : list
        A list of alpha values to use for the traces, in order; should match the
        length of the first dimension of dat. Default is 1.
    linewidths : list
        A list of line widths to use for the traces, in order; should match the
        length of the first dimension of dat. Default is 0.7.
    linestyles : list
        A list of line styles to use for the traces, in order; should match the
        length of the first dimension of dat. Default is a solid line.
    markers : list
        A list of markers to use for the traces, in order; should match the 
        length of the first dimension of dat. If no markers are given then only
        lines are plotted. If markers are wanted for some but not all traces,
        'None' should be used for the dat trace indices where lines are wanted.
        The '|' marker symbol is intended for plotting Boolean traces (e.g. 
        spike trains) and is implemented somewhat uniquely, with 0 values 
        converted to np.nan, 1 values plotted 85% of the way up the y-axis, and 
        different default values used for markeredgewidth and markersize. 
    markersizes : list
        A list of marker sizes to use for the traces, in order; should match the 
        length of the first dimension of dat. Default is 4.
    markeredgewidths : list
        A list of marker edge widths to use for the traces, in order; should
        match the length of the first dimension of dat. Default is 0.5.
    legend_labels : list
        A list of labels to use for the legend, in order; should be <= the 
        length of the first dimension of dat. If no legend_labels are given then
        no legend is made.    
    legend_size : int or float    
        Legend size, only relevant if legend labels are provided.
    figsize : list or tuple
        (x, y) size of the figure in inches.
    title : str
        Axis title.
    same_yaxis : bool
        If True, all subplots use the same ymin and ymax. Otherwise ylims
        are set optimally for each subplot.
        
    Returns
    -------
    fig, ax
        matplotlib figure and axis handles for further plot manipulation. 
        
    Examples
    --------
    freq = 4
    sampling_rate = 200
    duration = 2
    n_samples = sampling_rate * duration
    sin_wave = np.sin(np.linspace(-np.pi, np.pi, n_samples) * duration * freq)
    cos_wave = np.cos(np.linspace(-np.pi, np.pi, n_samples) * duration * freq)
    dat = np.vstack((sin_wave, cos_wave))
    
    # Plot the first 1 sec of data 
    fig, ax = plot_trace(dat, start=0, duration=1, x_units='secs', 
                         sampling_rate=sampling_rate, colors=['C0', 'C2'], 
                         markers=[None, '+'], 
                         legend_labels=['sin_wave', 'cos_wave'])                                         
    """
    if len(dat.shape) == 1:
        dat = np.expand_dims(dat, axis=0)
    n_traces = dat.shape[0]
    n_samples = dat.shape[1]
    if spike_dat is not None:
        spike_dat_ = spike_dat.copy()
        if len(spike_dat_.shape) == 1:
            spike_dat_ = np.expand_dims(spike_dat_, axis=0)
        n_units = spike_dat_.shape[0]
    else:
        n_units = 0
    
    # Convert units from volts to microvolts.
    dat = dat * 1e6
    
    # Subtract the mean from each channel.
    if sub_mean:
        dat = dat - np.expand_dims(np.mean(dat, axis=1), axis=1)
        
    # Spread out each channel on the y-axis.
    if create_spread:
        offsets = np.arange(n_traces) * 1.2 * (np.percentile(dat, 99) - np.percentile(dat, 1))
        dat = (dat + np.expand_dims(offsets, axis=-1))
        
    # Determine the x-axis units.
    if x_units == 'samples':
        if stop:
            stop = min(stop, n_samples)
        else:
            if duration:
                stop = min(start + duration, n_samples)
            else:
                stop = n_samples
        start = start
        if x_inds is None:
            x_inds = np.arange(n_samples) + 1
        if not x_lab:
            x_lab = 'Time (samples at {} Hz)'.format(sampling_rate)
    elif x_units == 'ms':
        if stop:
            stop = min(stop * (sampling_rate * 1e-3), n_samples)
        else:
            if duration:
                stop = min((start + duration) * (sampling_rate * 1e-3), n_samples)
            else:
                stop = n_samples
        start = start * (sampling_rate * 1e-3)
        if x_inds is None:
            x_inds = (np.arange(n_samples) + 1) / (sampling_rate * 1e-3)
        if not x_lab:
            x_lab = 'Time (ms)'
    elif x_units == 'secs':
        if stop:
            stop = min(stop * sampling_rate, n_samples)
        else:
            if duration:
                stop = min((start + duration) * sampling_rate, n_samples)
            else:
                stop = n_samples
        start = start * sampling_rate
        if x_inds is None:
            x_inds = (np.arange(n_samples) + 1) / sampling_rate
        if not x_lab:
            x_lab = 'Time (secs)'
    elif x_units == 'mins':
        if stop:
            stop = min(stop * (sampling_rate * 60), n_samples)
        else:
            if duration:
                stop = min((start + duration) * (sampling_rate * 60), n_samples)
            else:
                stop = n_samples
        start = start * (sampling_rate * 60)
        if x_inds is None:
            x_inds = (np.arange(n_samples) + 1) / (sampling_rate * 60)
        if not x_lab:
            x_lab = 'Time (mins)'
    start = int(start)
    stop = int(stop)
    
    # Split duration into nwin intervals.
    wins = np.linspace(start, stop, nwin+1, dtype=int)
    
    # Determine the y-axis limits.
    if same_yaxis:
        dat_min = np.min(dat[:, start:stop])
        dat_max = np.max(dat[:, start:stop])
        ymin = dat_min - (0.1 * abs(dat_min))
        ymax = dat_max + (0.1 * abs(dat_max))
        if ymin == 0:
            ymin = -0.1 * ymax
        if ymax == 0:
            ymax = 0.1 * -ymin
    else:
        ymin = []
        ymax = []
        for x in range(nwin):
            dat_min = np.min(dat[:, wins[x]:wins[x+1]])
            dat_max = np.max(dat[:, wins[x]:wins[x+1]])
            ymin_ = dat_min - (0.1 * abs(dat_min))
            ymax_ = dat_max + (0.1 * abs(dat_max))
            if ymin_ == 0:
                ymin_ = -0.1 * ymax
            if ymax_ == 0:
                ymax_ = 0.1 * -ymin
            ymin.append(ymin_)
            ymax.append(ymax_)
    
    # Create default lists for plot aesthetics that weren't passed as inputs.
    if not alphas:
        alphas = [1 for i in range(n_traces)]
    if not linewidths:
        linewidths = [0.7 for i in range(n_traces)]
    if not linestyles:
        linestyles = ['-' for i in range(n_traces)]
    if not markers: 
        markers = ['|' for i in range(n_units)]
    if not markersizes:
        markersizes = [12 for i in range(n_units)]
    if not markeredgewidths:
        markeredgewidths = [0.8 for i in range(n_units)]
    for i in range(n_units):
        if same_yaxis:
            set_val = ymax - ((ymax - ymin) * 0.15)
        else:
            x = ymax.index(min(ymax))
            set_val = ymax[x] - ((ymax[x] - ymin[x]) * 0.15)
        spike_dat_[i, spike_dat_[i, :]>0] = set_val
        spike_dat_[i, spike_dat_[i, :]==0] = np.nan
    if legend_labels and not legend_size:
        legend_size = 8
        
    # Create the plot.
    if fig is None and ax is None:     
        fig, ax = plt.subplots(nwin, 1, dpi=dpi, figsize=figsize)  
    ax = np.ravel(ax) 
    for x in range(len(ax)):
        start = wins[x]
        stop = wins[x+1]
        for i in range(n_traces):
            if colors:
                ax[x].plot(x_inds[start:stop][::step], 
                           dat[i, start:stop][::step], 
                           linewidth=linewidths[i], 
                           linestyle=linestyles[i], 
                           color=colors[i], 
                           alpha=alphas[i])
            else:
                ax[x].plot(x_inds[start:stop][::step], 
                           dat[i, start:stop][::step], 
                           linewidth=linewidths[i], 
                           linestyle=linestyles[i],
                           alpha=alphas[i])
        for i in range(n_units):
            if spike_colors:
                ax[x].plot(x_inds[start:stop][::step], 
                           spike_dat_[i, start:stop][::step], 
                           linewidth=0, 
                           marker=markers[i], 
                           markerfacecolor=spike_colors[i], 
                           markeredgecolor=spike_colors[i],
                           alpha=alphas[i], 
                           markersize=markersizes[i], 
                           markeredgewidth=markeredgewidths[i])
            else:
                ax[x].plot(x_inds[start:stop][::step], 
                           spike_dat_[i, start:stop][::step], 
                           linewidth=0, 
                           marker=markers[i], 
                           alpha=alphas[i], 
                           markersize=markersizes[i], 
                           markeredgewidth=markeredgewidths[i])
        
        if same_yaxis:        
            ax[x].axis([x_inds[start], x_inds[stop-1], ymin, ymax])
        else:
            ax[x].axis([x_inds[start], x_inds[stop-1], ymin[x], ymax[x]])
        ax[x].tick_params(labelsize=tickfontsize)
        ax[x].spines['right'].set_visible(False)
        ax[x].spines['top'].set_visible(False)
        if y_lab:
            ax[x].set_ylabel(y_lab, fontsize=labelfontsize, labelpad=8)
        else:
            ax[x].set_ylabel('μV', fontsize=labelfontsize, labelpad=8)
    
    ax[-1].set_xlabel(x_lab, fontsize=labelfontsize, labelpad=8)
    if legend_labels:
        ax[0].legend(legend_labels, loc=1, fontsize=tickfontsize, prop={'size': legend_size}, bbox_to_anchor=[1, 1.1])
    if title:
        fig.suptitle(title, fontsize=figfontsize)
    fig.tight_layout()
    
    return fig, ax