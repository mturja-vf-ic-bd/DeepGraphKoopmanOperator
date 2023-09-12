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
  return np.round(np.mean(smape[:, :fh // 3]),
                  3), np.round(np.mean(smape[:, fh // 3:fh // 3 * 2]),
                               3), np.round(np.mean(smape[:, -fh // 3:]),
                                            3), np.round(np.mean(smape), 3)
