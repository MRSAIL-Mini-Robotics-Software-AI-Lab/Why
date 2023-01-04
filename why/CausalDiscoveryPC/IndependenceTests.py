"""
Module for testing independence between variables
"""
import scipy
from sklearn.linear_model import LinearRegression


def compute_conditional_correlation(x, y, z):
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

def test_conditional_independence_pearsons(data, x, y, z):
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
    alpha = 0.05 # set significance level

    # Extract the relevant columns from the data
    x_values = data[x]
    y_values = data[y]

    if not z:
        _, pvalue_xy = scipy.stats.pearsonr(x_values, y_values)
        return pvalue_xy < alpha

    z_values = data[z]

    # Compute the Pearson's correlation coefficient between x and y given z
    _, pvalue_xy_given_z = compute_conditional_correlation(x_values, y_values, z_values)

    # If the conditional correlation is significantly different from the unconditional correlation
    # Return True, otherwise return False
    return pvalue_xy_given_z < alpha
