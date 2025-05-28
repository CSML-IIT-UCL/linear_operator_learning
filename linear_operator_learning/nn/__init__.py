"""Neural network methods entry point."""

import linear_operator_learning.nn.functional as functional
import linear_operator_learning.nn.stats as stats
from linear_operator_learning.nn.modules import *  # noqa: F403
from linear_operator_learning.nn.regressors import eig, evaluate_eigenfunction, ridge_least_squares
