import mltb

mltb.plot.twin_axes_timeseries_plot(
    [1, 2, 3, 3, 4, 2, 3], "a", [2, 1, 2, 1, 5], "b", shift_2=2, title="My Title", start_timestep_number=1
)
