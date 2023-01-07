"""
Module for testing independence between variables
"""
from typing import Protocol

import scipy
import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression


class IndependenceTest(Protocol):
    """
    Protocol for testing independence between variables
    """

    def __init__(self, alpha=0.05):
        ...

    def test_conditional_independence(self, data, x, y, z):
        """
        Test the hypothesis of conditional independence between x and y given z.
        """


class PearsonsCorrelation:
    """
    Class for testing independence between variables using Pearson's correlation coefficient
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def test_conditional_independence(self, data, x, y, z):
        """Test the hypothesis of conditional independence between x and y given z.

        Parameters:
        data: pandas DataFrame
            The data used to test the hypothesis.
        x: str
            The name of the first variable.
        y: str
            The name of the second variable.
        z: list of str
            The names of the conditioning variables.

        Returns:
        bool
            True if the hypothesis of conditional independence is rejected,
            False otherwise.
        """
        # Extract the relevant columns from the data
        x_values = data[x]
        y_values = data[y]

        if not z:
            _, pvalue_xy = scipy.stats.pearsonr(x_values, y_values)
            return pvalue_xy < self.alpha

        z_values = data[z]

        # Compute the Pearson's correlation coefficient between x and y given z
        _, pvalue_xy_given_z = self.compute_conditional_correlation(
            x_values, y_values, z_values
        )

        # If the conditional correlation is significantly different from the uncond correlation
        # Return True, otherwise return False
        return pvalue_xy_given_z < self.alpha

    def compute_conditional_correlation(self, x, y, z):
        """Compute the Pearson's correlation coefficient between x and y given z.
        Parameters:
        x: pandas Series
            The values of the first variable.
        y: pandas Series
            The values of the second variable.
        z: pandas DataFrame
            The values of the conditioning variables.

        Returns:
        float
            The Pearson's correlation coefficient between x and y given z.
        """

        # Fit a linear regression model to predict y from x and z
        model = LinearRegression()
        model.fit(z, y)

        # Compute the residuals of the model
        residuals = y - model.predict(z)

        # Compute the Pearson's correlation coefficient between x and the residuals
        rxy_given_z, pvalue_xy_given_z = scipy.stats.pearsonr(x, residuals)

        return rxy_given_z, pvalue_xy_given_z


class PartialCorrelation:
    """
    Class for testing independence between variables using partial correlation
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def test_conditional_independence(self, data, x, y, z):
        """Test the hypothesis of conditional independence between x and y given z.

        Parameters:
        data: pandas DataFrame
            The data used to test the hypothesis.
        x: str
            The name of the first variable.
        y: str
            The name of the second variable.
        z: list of str
            The names of the conditioning variables.

        Returns:
        bool
            True if the hypothesis of conditional independence is rejected,
            False otherwise.
        """
        target = np.array(data[x])
        predictors = np.array(data[[y] + z])

        # add a constant term to the model
        predictors = sm.add_constant(predictors)
        model = sm.OLS(target, predictors).fit()

        # retrieve the p-value of y
        p_value = model.pvalues[1]

        # Return True if the hypothesis is rejected, False otherwise.
        return p_value <= self.alpha


class ChiSquared:
    """
    Class for testing independence between variables using chi-squared test
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def test_conditional_independence(self, data, x, y, z):
        """Test the hypothesis of conditional independence between x and y given z.

        Parameters:
        data: pandas DataFrame
            The data used to test the hypothesis.
        x: str
            The name of the first variable.
        y: str
            The name of the second variable.
        z: list of str
            The names of the conditioning variables.

        Returns:
        bool
            True if the hypothesis of conditional independence is rejected,
            False otherwise.
        """
        if z:
            # Separate dataframe
            grouped_data = data.groupby(z)

            for group in grouped_data:
                # Compute the observed frequencies of each combination of x, y, and z values
                observed_frequencies = pd.crosstab(
                    index=group[1][x], columns=group[1][y]
                )
                # Calculate the chi-squared statistic and p-value
                _, p_val, _, _ = chi2_contingency(observed_frequencies)

                if p_val < self.alpha:
                    return True  # reject the hypothesis of conditional independence
        else:
            # Compute the observed frequencies of each combination of x, y, and z values
            observed_frequencies = pd.crosstab(index=data[x], columns=data[y])

            # Calculate the chi-squared statistic and p-value
            _, p_val, _, _ = chi2_contingency(observed_frequencies)

            # Determine if the hypothesis of conditional independence should be rejected
            if p_val < self.alpha:
                return True  # reject the hypothesis of conditional independence

        return False  # fail to reject the hypothesis of conditional independence
