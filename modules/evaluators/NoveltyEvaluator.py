# # TODO 高速化
#
# import math
#
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
#
# from ..models import BaseEmbeddingModel
# from .BaseEvaluator import BaseEvaluator
#
#
# class NoveltyEvaluator(BaseEvaluator):
#     """Class of evaluators computing Novelty for each user and calculating average."""
#
#     def __init__(
#             self, test_set: torch.Tensor, interaction_set: np.ndarray, ks: list = [5]
#     ):
#         """Set test data and metrics.
#
#         Args:
#             test_set (torch.Tensor): test data which column is [user_id, item_id, rating].
#             interaction_set (np.ndarray): user_id, item_id which the user has already rated pair [user_id, item_id]
#             ks (int, optional): A list of @k. Defaults to [5].
#
#         arguments of each function must be
#             y_hat_user (np.ndarray) : prediction of relevance
#             k : a number of top item considered.
#         """
#         super().__init__(test_set)
#
#         self.interaction_set = interaction_set
#         self.ks = ks
#
#     def eval_user(self, model: BaseEmbeddingModel, uid: int, user_ids: torch.Tensor) -> pd.DataFrame:
#         """Method of evaluating for given user.
#
#         Args:
#             model (BaseEmbeddingModel): models which have user and item embeddings.
#             uid (int): user id
#             user_ids (torch.Tensor): all user id in test set
#
#         Returns:
#             (pd.DataFrame): a row of DataFrame which has scores for each metrics and k for the user.
#         """
#
#         score_dict = {}
#
#         # Extract pairs of the target user and items from the test data
#         user_indices = self.test_set[:, 0] == uid
#         test_set_pair = self.test_set[user_indices, :2]
#         # Get the predicted distances of the target user and items
#         y_hat_user = model.predict(test_set_pair).to("cpu").detach().numpy()
#
#         for k in self.ks:
#             eval_name = f"novelty@{k}"
#
#             # Get the IDs of the top k items
#             pred_topk_indices = (-y_hat_user).argsort()[:k]
#             topk_item_ids = self.test_set[pred_topk_indices, 1]
#
#             sum_log_p = 0
#             for item_id in topk_item_ids:
#                 sum_log_p += log_p(user_ids, item_id, self.interaction_set)
#
#             novelty_score = sum_log_p / k
#             score_dict[eval_name] = [novelty_score]
#
#         df_eval_sub = pd.DataFrame(score_dict)
#
#         return df_eval_sub
#
#     def score(
#             self, model: BaseEmbeddingModel, reduction="mean", verbose=True
#     ) -> pd.DataFrame:
#         """Method of calculating average novelty for all users.
#
#         Args:
#             model (BaseEmbeddingModel): models which have user and item embeddings.
#             reduction (str, optional): reduction method. Defaults to "mean".
#             verbose (bool, optional): displaying progress bar or not during evaluating. Defaults to True.
#
#         Returns:
#             pd.DataFrame: a row of DataFrame which has average scores
#         """
#
#         users = torch.unique(self.test_set[:, 0])
#         df_eval = pd.DataFrame({f"novelty@{k}": [] for k in self.ks})
#
#         if verbose:
#             for uid in tqdm(users):
#                 df_eval_sub = self.eval_user(model, uid, users)
#                 df_eval = pd.concat([df_eval, df_eval_sub])
#         else:
#             for uid in users:
#                 df_eval_sub = self.eval_user(model, uid, users)
#                 df_eval = pd.concat([df_eval, df_eval_sub])
#
#         if reduction == "mean":
#             score = pd.DataFrame(df_eval.mean(axis=0)).T
#
#         else:
#             score = df_eval.copy()
#
#         return score
#
#
# def imp(user_id, item_id, interaction_set):
#     return int(np.any(np.all(interaction_set == np.array([user_id, item_id]), axis=1)))
#
#
# def log_p(user_ids, item_id, interaction_set):
#     sum_imp = 0
#     for user_id in user_ids:
#         sum_imp += imp(user_id, item_id, interaction_set)
#
#     return -math.log(sum_imp / len(user_ids), 2)
