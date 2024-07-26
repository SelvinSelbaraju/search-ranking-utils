from typing import Optional, Tuple
import numpy as np
import pandas as pd


class PopularityBaseline:
    """
    Output the unpersonalised target probability
    Stores a lookup table of item ids and target probability
    If id not found, optionall specify default
    Default not specified, use global target probability
    """

    def __init__(
        self,
        target_col: str,
        item_id_col: str,
        default_val: Optional[float] = None,
    ):
        self.target_col = target_col
        self.item_id_col = item_id_col
        self.default_val = default_val

    def fit(self, X: pd.DataFrame) -> None:
        self.score_lookup = {}
        scores = X.groupby(self.item_id_col)[self.target_col].mean()
        for i in range(len(scores)):
            item = scores.index[i]
            prob = scores[i]
            self.score_lookup[item] = prob
        # Use global prob
        if not self.default_val:
            self.default_val = X[self.target_col].mean()

    def _get_score(self, item_id: str) -> float:
        return self.score_lookup.get(item_id, self.default_val)

    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return [p(target=0), p(target=1)] to match sklearn
        """
        scores = X[self.item_id_col].apply(lambda x: self._get_score(x))
        return 1 - scores.values, scores.values
