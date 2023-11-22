import numpy as np


def SMAPE(test_preds, test_tgts):
  """Metric for M4 dataset.

  Refer to
  https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py

  Args:
    test_preds: models' predictions with shape of
    (num_samples, forecasting horizon, ...)
    test_tgts: ground truth that has the same shape as test_preds.

  Returns:
    short, medium, long forecasting horizon prediction sMAPE.
  """
  smape = np.abs(test_preds - test_tgts) * 200 / (
      np.abs(test_preds) + np.abs(test_tgts))
  fh = test_preds.shape[1]
  # return short, medium, long forecasting horizon and total sMAPE
  return np.round(np.mean(smape[:, :fh // 4]),
                  3), np.round(np.mean(smape[:, fh // 4:fh // 4 * 2]),
                               3), np.round(np.mean(smape[:, fh // 4 * 2: fh // 4 * 3]),
                                            3), np.round(np.mean(smape[:, -fh // 4: ]),
                                                         3), np.round(np.mean(smape), 3)

def WRMSE(
    test_preds,
    test_tgts,  # ground truth that has the same shape as test_preds
    weights=np.array([
        4.30406509320417, 6.779921907472252, 2.3978952727983707,
        4.406719247264253, 3.555348061489413, 1.3862943611198906,
        5.8944028342648505, 2.079441541679836, 1.0986122886681098,
        2.3978952727983707, 1.0986122886681098, 1.6094379124341005,
        2.079441541679836, 1.791759469228055
    ])  # Importance weights for 14 Cryptos
):
  """Metric for Cryptos return predictions.

  RMSE should be weighted by the importance of each stock
  Refer to https://www.kaggle.com/competitions/g-research-crypto-forecasting/
  data?select=asset_details.csv

  Args:
    test_preds: models' predictions with shape of (number of stocks,
    number of samples for each stock, forecasting horizons, 8 features)
    test_tgts: ground truth that has the same shape as test_preds.
    weights: Importance weights for 14 Cryptos

  Returns:
    short, medium, long forecasting horizon prediction Weighted RMSE.
  """

  weights = np.expand_dims(weights, axis=(1, 2, 3))

  fh = test_preds.shape[2]
  wrmse = ((test_preds - test_tgts) * weights)**2

  # only evaluate predictions based on the last feature
  # (15-min ahead residulized returns)
  # return short, medium, long forecasting horizon and total Weighted RMSE
  return (
      np.sqrt(np.mean(wrmse[Ellipsis, : fh // 3, -1])),
      np.sqrt(np.mean(wrmse[Ellipsis, fh // 3 : fh // 3 * 2, -1])),
      np.sqrt(np.mean(wrmse[Ellipsis, -fh // 3, -1])),
      np.sqrt(np.mean(wrmse[Ellipsis, -1])),
  )