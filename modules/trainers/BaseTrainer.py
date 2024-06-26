from typing import Optional
import torch
from torch import optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..evaluators.BaseEvaluator import BaseEvaluator
from ..losses.BaseLoss import BaseLoss
from ..losses.MinorTripletLoss import MinorTripletLoss
from ..models.BaseEmbeddingModel import BaseEmbeddingModel
from ..samplers.BaseSampler import BaseSampler
from ..searches import BaseSearch


class BaseTrainer:
    """Class of abstract trainers for recommend system in implicit feedback setting."""

    def __init__(
        self,
        model: BaseEmbeddingModel,
        optimizer: optim,
        criterion: BaseLoss,
        sampler: BaseSampler,
        no_progressbar: bool = False,
        column_names: Optional[dict] = None,
    ):
        """Set components for learning recommend system.

        Args:
            model (BaseEmbeddingModel): embedding models
            optimizer (optim): pytorch optimizer
            criterion (Union[BasePairwiseLoss, BaseTripletLoss]): losses function
            sampler (BaseSampler): samplers
            no_progressbar (bool, optional): displaying progress bar or not during evaluating. Defaults to False.
            column_names (Optional[dict]): samplers
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.sampler = sampler
        self.no_progressbar = no_progressbar
        self.valid_scores = None

        if column_names is not None:
            self.column_names = column_names
        else:
            self.column_names = {"user_id": 0, "item_id": 1}

    def fit(
        self,
        n_batch: int = 500,
        n_epoch: int = 10,
        search: Optional[BaseEvaluator] = None,
        valid_evaluator: Optional[BaseEvaluator] = None,
        valid_per_epoch: int = 5,
    ):

        # set evaluators and log dataframe
        valid_or_not = search is not None and valid_evaluator is not None
        if valid_or_not:
            self.valid_scores = valid_evaluator.score(search, no_progressbar=self.no_progressbar)
            self.valid_scores["epoch"] = 0
            self.valid_scores["losses"] = np.nan

        # start training
        for ep in range(n_epoch):
            accum_loss = 0

            # start epoch
            with tqdm(range(n_batch), total=n_batch, disable=self.no_progressbar) as pbar:
                for b in pbar:
                    # batch sampling
                    batch = self.sampler.get_pos_batch()
                    users = batch[:, self.column_names["user_id"]].reshape(-1, 1)
                    pos_items = batch[:, self.column_names["item_id"]].reshape(-1, 1)

                    if self.sampler.two_stage:
                        neg_candidates = self.sampler.get_and_set_candidates()
                        dist = self.model.spreadout_distance(pos_items, neg_candidates)
                        self.sampler.set_candidates_weight(dist, self.model.n_dim)

                    neg_items = self.sampler.get_neg_batch(users.reshape(-1))

                    # initialize gradient
                    self.model.zero_grad()

                    # compute distance
                    embeddings_dict = self.model(users, pos_items, neg_items)

                    # compute losses
                    if isinstance(self.criterion, MinorTripletLoss):
                        # Miner CML
                        embeddings_dict["pos_items"] = pos_items
                        loss = self.criterion(embeddings_dict, batch, self.column_names)
                    else:
                        loss = self.criterion(embeddings_dict, batch, self.column_names)

                    # adding losses for domain adaptation
                    if self.model.user_adaptor is not None:
                        loss += self.model.user_adaptor(
                            users, embeddings_dict["user_embedding"]
                        )
                    if self.model.item_adaptor is not None:
                        loss += self.model.item_adaptor(
                            pos_items, embeddings_dict["pos_item_embedding"]
                        )
                        loss += self.model.item_adaptor(
                            neg_items, embeddings_dict["neg_item_embedding"]
                        )

                    accum_loss += loss.item()

                    # gradient of losses
                    loss.backward()

                    # update models parameters
                    self.optimizer.step()

                    pbar.set_description_str(
                        f"epoch{ep+1} avg_loss:{accum_loss / (b+1) :.3f}"
                    )

            # compute metrics for epoch
            if valid_or_not and (
                ((ep + 1) % valid_per_epoch == 0) or (ep == n_epoch - 1)
            ):
                valid_scores_sub = valid_evaluator.score(search, no_progressbar=self.no_progressbar)
                valid_scores_sub["epoch"] = ep + 1
                valid_scores_sub["losses"] = accum_loss / n_batch
                self.valid_scores = pd.concat([self.valid_scores, valid_scores_sub])

    def valid(
            self,
            search: BaseSearch,
            valid_evaluator: Optional[BaseEvaluator]
    ):
        self.valid_scores = valid_evaluator.score(search, no_progressbar=self.no_progressbar)
