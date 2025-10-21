import time
import torch
import torch.nn as nn
import numpy as np

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
def RSE(pred, true):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0))**2 * (pred - pred.mean(0))**2).sum(0))
    return (u / d).mean()

def Corr(pred, true):
    sig_p = torch.std(pred, dim=0)
    sig_g = torch.std(true, dim=0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    ind = sig_g != 0
    corr = ((pred - m_p) * (true - m_g)).mean(0) / (sig_p * sig_g)
    corr = corr[ind].mean()
    return corr

def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))

def MSE(pred, true):
    return torch.mean((pred - true) ** 2)

def RMSE(pred, true):
    return torch.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return torch.mean(torch.abs((pred - true) / true))

def MSPE(pred, true):
    return torch.mean(((pred - true) / true) ** 2)

def test(model, test_loader, scaler_y, device):
  mae_loss = nn.L1Loss()
  running_loss = 0.0
  total_samples = 0

  model.eval()
  for batch_x, batch_y in test_loader:
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    b = batch_x.size(0)
    total_samples += b

    y_pred, _ = model(batch_x)

    y_true_pred = scaler_y.inverse_transform(y_pred.cpu().detach().numpy())
    y_true_true = scaler_y.inverse_transform(batch_y.cpu().detach().numpy())

    test_mse_loss = mae_loss(torch.from_numpy(y_true_pred), torch.from_numpy(y_true_true))
    running_loss += test_mse_loss.item() * b

  test_mae_loss = running_loss / len(test_loader.dataset)

  return test_mae_loss

def test_rse_corr(model, test_loader, dm, device):
    running_loss_metric_1 = 0.0
    running_loss_metric_2 = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            b = batch_x.size(0)
            total_samples += b

            y_pred, _ = model(batch_x)

            y_pred_np = y_pred.detach().cpu().numpy()
            y_true_np = batch_y.detach().cpu().numpy()

            y_pred_inv = dm.inverse_scale(y_pred_np)
            y_true_inv = dm.inverse_scale(y_true_np)

            y_pred_tensor = torch.from_numpy(y_pred_inv).float()
            y_true_tensor = torch.from_numpy(y_true_inv).float()

            test_metric_1 = RSE(y_pred_tensor, y_true_tensor)
            test_metric_2 = CORR(y_pred_tensor, y_true_tensor)
            running_loss_metric_1 += test_metric_1 * b
            running_loss_metric_2 += test_metric_2 * b

    test_metric_1_loss = running_loss_metric_1 / total_samples
    test_metric_2_loss = running_loss_metric_2 / total_samples
    return test_metric_1_loss, test_metric_2_loss

def test_mae_mse(model, test_loader, dm, device):
    running_loss_metric_1 = 0.0
    running_loss_metric_2 = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            b = batch_x.size(0)
            total_samples += b

            y_pred, _ = model(batch_x)

            y_pred_np = y_pred.detach().cpu().numpy()
            y_true_np = batch_y.detach().cpu().numpy()

            y_pred_inv = dm.inverse_scale(y_pred_np)
            y_true_inv = dm.inverse_scale(y_true_np)

            y_pred_tensor = torch.from_numpy(y_pred_inv).float()
            y_true_tensor = torch.from_numpy(y_true_inv).float()

            test_metric_1 = MAE(y_pred_tensor, y_true_tensor)
            test_metric_2 = MSE(y_pred_tensor, y_true_tensor)
            running_loss_metric_1 += test_metric_1 * b
            running_loss_metric_2 += test_metric_2 * b

    test_metric_1_loss = running_loss_metric_1 / total_samples
    test_metric_2_loss = running_loss_metric_2 / total_samples
    return test_metric_1_loss, test_metric_2_loss

def train_val(model, criterion, optimizer, train_loader, val_loader, device, batch_size=128, epochs=10):
  train_loss_per_epoch = []
  val_loss_per_epoch = []

  torch.cuda.synchronize()
  start_time = time.time()
  for epoch in range(epochs):
      model.train()
      running_loss = 0.0
      total_samples = 0
      for i, (batch_x, batch_y) in enumerate(train_loader):
          batch_x = batch_x.float().to(device) # (128, 5, 20)
          batch_y = batch_y.float().to(device) # (128, 4)

          if epoch == 0 and i == 0:
              print(f"batch_x shape: {batch_x.size()}")
              print(f"batch_y shape: {batch_y.size()}")

          b = batch_x.size(0)
          total_samples += b
          h0 = None

          # pred
          y_pred, _ = model(batch_x, h0)

          # backprop
          loss = criterion(y_pred, batch_y)
          running_loss += loss.item() * b
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

      train_epoch_loss = running_loss / total_samples
      train_loss_per_epoch.append(train_epoch_loss)


      model.eval()
      running_loss = 0.0
      total_samples = 0
      for batch_x, batch_y in val_loader:
          batch_x = batch_x.float().to(device) # (128, 5, 20)
          batch_y = batch_y.float().to(device) # (128, 4, 1)
          b = batch_x.size(0)
          total_samples += b
          h0 = None

          # pred
          y_pred, _ = model(batch_x)
          loss = criterion(y_pred, batch_y)
          running_loss += loss.item() * b

      val_epoch_loss = running_loss / total_samples
      val_loss_per_epoch.append(val_epoch_loss)

      print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

  torch.cuda.synchronize()
  total_time = time.time() - start_time

  # plt.plot(train_loss_per_epoch)
  # plt.plot(val_loss_per_epoch)
  # plt.legend(['train', 'val'])
  # plt.show()

  return model, total_time, train_loss_per_epoch, val_loss_per_epoch


# for keras
def evaluate_loader(model, test_loader, scaler_y):
  all_y_true = []
  all_y_pred = []
  for X, y in test_loader:
    y_pred = model(X, training=False)
    all_y_true.append(y)
    all_y_pred.append(y_pred)

  if len(all_y_true) == 0:
    return np.nan

  all_y_true = tf.concat(all_y_true, axis=0).numpy()
  all_y_pred = tf.concat(all_y_pred, axis=0).numpy()

  # inverse transform to original scale
  y_true_orig = scaler_y.inverse_transform(all_y_true)
  y_pred_orig = scaler_y.inverse_transform(all_y_pred)

  mae = mean_absolute_error(y_true_orig, y_pred_orig)
  return mae