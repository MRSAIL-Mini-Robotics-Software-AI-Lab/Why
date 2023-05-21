'''
Sensitivity analysis for a linear single confounder model
'''
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearSingleConfounder:
    '''
    Class for sensitivity analysis for a linear single confounder model
    '''
    def __init__(self, data_frame:pd.DataFrame):
        self.data_frame = data_frame

    def sensitivity_contour(self, estimate:float,
                            contour_values:Optional[List[float]]=None,
                            x_ranges:List[float]=[-5, 5],
                            y_ranges:List[float]=[-5, 5],
                            region_range:Optional[List[float]]=None) -> None:
        '''
        Draw sensitivity contour for a linear single confounder model

        Parameters
        ----------
        estimate : float
            The estimated ATE
        contour_values : List[float], by default None
            Values to plot on the contour, the default is [estimate/2, 0, -1*estimate]
            These are the values where estimate - beta/alpha = contour_value
        x_ranges : List[float], by default [-5, 5]
            The x axis range for the plot
        y_ranges : List[float], by default [-5, 5]
            The y axis range for the plot
        region_range : List[float], by default None
            The range of ate values to shade in the plot
        '''
        if contour_values is None:
            contour_values = []
            contour_values.append(estimate/2)
            contour_values.append(0)
            contour_values.append(-1*estimate)

        plt.xlim([*x_ranges])
        plt.ylim([*y_ranges])
        plt.grid()
        plt.scatter([0],[0], label=f"ate={estimate:.2f}")

        x_vals = np.linspace(*x_ranges, 100)
        i = np.where(np.abs(x_vals)<0.0001)[0]
        x_vals = np.delete(x_vals, i)
        for val in contour_values:
            y_vals = (estimate - val)/ x_vals
            plt.plot(x_vals, y_vals, label=f"ate = {val:.2f}")

        if region_range is not None:
            self.shade_region(estimate, x_vals, region_range)

        plt.ylabel("βu")
        plt.xlabel("1/αu")
        plt.legend()

    def shade_region(self, estimate:float, x_vals:np.array, region_range:List[float]) -> None:
        '''
        Shade the region of the sensitivity contour

        Parameters
        ----------
        estimate : float
            The estimated ATE
        x_vals : np.array
            The x values for the sensitivity contour
        region_range : List[float]
            The range of ate values to shade in the plot
        '''
        assert region_range[0] < region_range[1]
        y_upper = (estimate - region_range[1])/ x_vals
        y_lower = (estimate - region_range[0])/ x_vals
        plt.fill_between(x_vals, y_upper, y_lower, alpha=0.2)
