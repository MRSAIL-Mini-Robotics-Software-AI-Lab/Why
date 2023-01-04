'''
Methods for computing ate bounds using different assumptions
Also contains enum for the different assumptions
'''
import enum
from typing import Tuple, List

import numpy as np
import pandas as pd


class Bounds(enum.Enum):
    '''
    Enum for the different bound assumptions possible
    '''
    NonNegativeMonotoneTreatmentResponse = 1
    NonPositiveMonoToneTreatmentResponse = 2
    MonotoneTreatmentSelection = 3
    OptimalTreatmentSelection = 4

class BoundsEstimator:
    '''
    Class for computing ate bounds using different assumptions
    '''
    def __init__(self, data_frame:pd.DataFrame):
        self.data_frame = data_frame

    def compute_bound(self, treatment:str,
                            outcome:str,
                            assumptions:List[int]) -> Tuple[float, float]:
        '''
        Compute the ate bounds using the given assumptions

        Parameters
        ----------
        treatment : str
            Variable name of the treatment column
        outcome : str
            Variable name of the outcome column
        assumptions : List[int]
            List of assumptions to use, available in the Bounds enum

        Returns
        -------
        Tuple[float, float]
            The lower and upper bounds of the ate
        '''
        if Bounds.NonNegativeMonotoneTreatmentResponse in assumptions:
            assert Bounds.NonPositiveMonoToneTreatmentResponse not in assumptions

        lower_bound, upper_bound = self.no_assumptions_bound(treatment, outcome)
        if Bounds.NonNegativeMonotoneTreatmentResponse in assumptions:
            nonneg_lower, nonneg_upper = self.monotone_treatment_response(treatment, outcome, True)
            upper_bound = min(nonneg_upper, upper_bound)
            lower_bound = max(nonneg_lower, lower_bound)
        if Bounds.NonPositiveMonoToneTreatmentResponse in assumptions:
            nonpos_lower, nonpos_upper = self.monotone_treatment_response(treatment, outcome, False)
            upper_bound = min(nonpos_upper, upper_bound)
            lower_bound = max(nonpos_lower, lower_bound)
        if Bounds.MonotoneTreatmentSelection in assumptions:
            mts_lower, mts_upper = self.monotone_treatment_selection(treatment, outcome)
            upper_bound = min(mts_upper, upper_bound)
            lower_bound = max(mts_lower, lower_bound)
        if Bounds.OptimalTreatmentSelection in assumptions:
            ots_lower, ots_upper = self.optimal_treatment_selection(treatment, outcome)
            upper_bound = min(ots_upper, upper_bound)
            lower_bound = max(ots_lower, lower_bound)

        return lower_bound, upper_bound

    def no_assumptions_bound(self, treatment:str, outcome:str) -> Tuple[float, float]:
        '''
        Compute the ate bounds using no assumptions

        Parameters
        ----------
        treatment : str
            Variable name of the treatment column
        outcome : str
            Variable name of the outcome column

        Returns
        -------
        lower_bound, upper_bound : Tuple[float]
            The lower and upper bounds of the ate
        '''
        treatment_col = self.data_frame[treatment]
        outcome_col = self.data_frame[outcome]

        a, b = 0, 1
        p_treatment = np.sum(treatment_col) / len(treatment_col)
        p_outcome_given_t1 = np.sum(outcome_col[treatment_col == 1]) / np.sum(treatment_col)
        p_outcome_given_t0 = np.sum(outcome_col[treatment_col == 0]) / np.sum(treatment_col == 0)

        common_term = p_treatment*p_outcome_given_t1 - (1-p_treatment)*p_outcome_given_t0
        upper_bound =  (1-p_treatment)*b - p_treatment*a + common_term
        lower_bound = (1-p_treatment)*a - p_treatment*b + common_term

        return lower_bound, upper_bound

    def monotone_treatment_response(self, treatment:str,
                                            outcome:str,
                                            nonnegative:bool) -> Tuple[float, float]:
        '''
        Compute the ate bounds using the monotone treatment response assumption
        For the nonnegative case:
            Assume treatment always helps [for all outcome(treatment=1) >= outcome(treatment=0)]
        The nonpositive case is the opposite

        Parameters
        ----------
        treatment : str
            Variable name of the treatment column
        outcome : str
            Variable name of the outcome column
        nonnegative : bool
            If True, will use the nonnegative monotone treatment response assumption
            else, will use the nonpositive monotone treatment response assumption

        Returns
        -------
        lower_bound, upper_bound : Tuple[float]
            The lower and upper bounds of the ate
        '''
        lower_bound, upper_bound = self.no_assumptions_bound(treatment, outcome)
        if nonnegative:
            lower_bound = 0
        else:
            upper_bound = 0
        return lower_bound, upper_bound

    def monotone_treatment_selection(self, treatment:str, outcome:str) -> Tuple[float, float]:
        '''
        Compute the ate bounds using the monotone treatment selection assumption

        Parameters
        ----------
        treatment : str
            Variable name of the treatment column
        outcome : str
            Variable name of the outcome column

        Returns
        -------
        Tuple[float, float]
            The lower and upper bounds of the ate
        '''
        treatment_col = self.data_frame[treatment]
        outcome_col = self.data_frame[outcome]

        p_outcome_given_t1 = np.sum(outcome_col[treatment_col == 1]) / np.sum(treatment_col)
        p_outcome_given_t0 = np.sum(outcome_col[treatment_col == 0]) / np.sum(treatment_col == 0)

        lower_bound, upper_bound = self.no_assumptions_bound(treatment, outcome)
        upper_bound = min(upper_bound, p_outcome_given_t1 - p_outcome_given_t0)

        return lower_bound, upper_bound

    def optimal_treatment_selection(self, treatment:str, outcome:str) -> Tuple[float, float]:
        '''
        Compute the ate bounds using the optimal treatment selection assumption

        Parameters
        ----------
        treatment : str
            Variable name of the treatment column
        outcome : str
            Variable name of the outcome column

        Returns
        -------
        Tuple[float, float]
            The lower and upper bounds of the ate
        '''
        treatment_col = self.data_frame[treatment]
        outcome_col = self.data_frame[outcome]

        a = 0
        p_treatment = np.sum(treatment_col) / len(treatment_col)
        p_outcome_given_t1 = np.sum(outcome_col[treatment_col == 1]) / np.sum(treatment_col)
        p_outcome_given_t0 = np.sum(outcome_col[treatment_col == 0]) / np.sum(treatment_col == 0)

        upper_bound_1 = p_treatment*p_outcome_given_t1 - p_treatment*a
        lower_bound_1 = (1-p_treatment)*a - (1-p_treatment)*p_outcome_given_t0

        upper_bound_2 = p_outcome_given_t1 - p_treatment*a - (1-p_treatment)*p_outcome_given_t0
        lower_bound_2 = p_treatment*p_outcome_given_t1 + (1-p_treatment)*a - p_outcome_given_t0

        upper_bound = min(upper_bound_1, upper_bound_2)
        lower_bound = max(lower_bound_1, lower_bound_2)

        return lower_bound, upper_bound
