#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss, Metrics and Callback functions to use for model

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as cbck
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.metrics import roc_auc_score as roc

# ----------------------------------------------------------------------------------
"Utility Functions and Global Params"

LOGS_DIR = "experiments/CAMELOT/"


def tf_log(tensor):
    return tf.math.log(tensor + 1e-8)


def tf_divide(tensor1, tensor2):
    return tf.math.divide_no_nan(tensor1, tensor2)


def np_log(array):
    return np.log(array + 1e-8)


# ------------------------------------------------------------------------------------
"""Loss Functions"""


def l_crit(y_true, y_pred, name='pred_clus_loss'):
    """
    Negative weighted predictive clustering loss. Computes Cross-entropy between categorical y_true and y_pred.
    This is minimised when y_pred matches y_true.

    Params:
    - y_true: array-like of shape (bs, num_outcs) of one-hot encoded true class.
    - y_pred: array-like of shape (bs, num_outcs) of probability class predictions.
    - name: name to give to operation.

    Returns:
    - loss_value: score indicating corresponding loss.
    """

    # Compute batch
    batch_neg_ce = - tf.reduce_sum(y_true * tf_log(y_pred))

    # Average over batch
    loss_value = tf.reduce_mean(batch_neg_ce, name=name)

    return loss_value


def l_phens(cluster_reps, name='phenotype_separation_loss'):
    """Cluster phenotype separation loss. Computes negative KL divergence between phenotypes summed over pairs of
    cluster representation vectors. This loss is minimised as cluster vectors are separated.

    Params:
    - cluster_reps: array-like of shape (K, latent_dim) of cluster representation vectors.
    - name: name to give to operation.

    Returns:
    - norm_loss: score indicating corresponding loss.
    """

    # Expand input to allow broadcasting
    embedding_column = tf.expand_dims(cluster_reps, axis=1)  # shape (K, 1, latent_dim)
    embedding_row = tf.expand_dims(cluster_reps, axis=0)  # shape (1, K, latent_dim)

    # Compute pairwise Euclidean distance between cluster vectors, and sum over pairs of clusters.
    pairwise_loss = - tf.reduce_sum((embedding_column * tf_log(tf_divide(embedding_column, embedding_row))), axis=-1)
    loss = tf.reduce_sum(pairwise_loss, axis=None, name=name) - 1e-8

    # normalise by factor
    K = cluster_reps.get_shape()[0]
    norm_loss = loss / (K * (K - 1))

    return norm_loss


def l_prob(clusters_prob):
    """
    Cluster assignment confidence loss. Computes entropy of cluster distribution probability values.
    This is minimised when the cluster distribution is a delta dirac distribution (i.e. pinpoints to a single cluster).

    Params:
    - clusters_prob: array-like of shape (bs, K) of cluster_assignments distributions.

    Returns:
    - loss_value: score indicating corresponding loss.
    """

    # Compute entropy of dist per sample
    entropy = - clusters_prob * tf_log(clusters_prob)

    # Compute negative entropy
    batch_loss = tf.reduce_sum(entropy, axis=0)

    return batch_loss


# ----------------------------------------------------------------------------------
"Callback methods to update training procedure."


class CEClusSeparation(cbck.Callback):
    """
    Callback method to print Normalised Cross-Entropy separation between cluster phenotypes.
    Higher values indicate higher separation (which is good!)

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, *args, **kwargs):

        # Print information if matches interval epoch length
        if epoch % self.interval == 0:

            # Initialise callback value, and determine K
            cbck_value, K = 0, self.model.K
            clus_phenotypes = self.model.compute_cluster_phenotypes()

            # Iterate over all pairs of clusters and compute symmetric CE
            for i in range(K):
                for j in range(i + 1, K):
                    cbck_value += - np.sum(clus_phenotypes[i, :] * np_log(clus_phenotypes[j, :]))
                    cbck_value += - np.sum(clus_phenotypes[j, :] * np_log(clus_phenotypes[i, :]))

            # normalise and print output
            cbck_value = cbck_value / (K * (K + 1))

            print("End of Epoch {:d} - CE sep : {:.4f}".format(epoch, cbck_value))


class ConfusionMatrix(cbck.Callback):
    """
    Callback method to print Confusion Matrix over data.

    Output is a matrix indicating the amount of patients assigned to a target class and with a certain true class.

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

        # Compute number of outcomes
        self.C = self.y_val.shape[-1]

    def on_epoch_end(self, epoch, *args, **kwargs):

        # Print information if matches interval epoch length
        if epoch % self.interval == 0:

            # Initialise output Confusion matrix
            cm_output = np.zeros(shape=(self.C, self.C))

            # Compute prediction and true values in categorical format.
            y_pred = (self.model(self.X_val)).numpy()
            class_pred = np.argmax(y_pred, axis=-1)
            class_true = np.argmax(self.y_val, axis=-1)

            # Iterate through classes
            for true_class in range(self.C):
                for pred_class in range(self.C):
                    num_samples = np.logical_and(class_pred == pred_class, class_true == true_class).sum()
                    cm_output[true_class, pred_class] = num_samples

            # Print as pd.dataframe
            index = [f"TC{class_}" for class_ in range(1, self.C + 1)]
            columns = [f"PC{class_}" for class_ in range(1, self.C + 1)]

            cm_output = pd.DataFrame(cm_output, index=index, columns=columns)

            print("End of Epoch {:d} - Confusion matrix: \n {}".format(epoch, cm_output.astype(int)))


class AUROC(cbck.Callback):
    """
    Callback method to display AUROC value for predicted y.

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, *args, **kwargs):
        if epoch % self.interval == 0:
            # Compute predictions
            y_pred = self.model(self.X_val).numpy()

            # Compute ROC
            roc_auc_score = roc(y_true=self.y_val, y_score=y_pred,
                                average=None, multi_class='ovr')

            print("End of Epoch {:d} - OVR ROC score: {}".format(epoch, roc_auc_score))


class PrintClusterInfo(cbck.Callback):
    """
    Callback method to display cluster distribution information assignment.

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, *args, **kwargs):
        if epoch % self.interval == 0:

            # Compute cluster_predictions
            clus_pred = self.model.compute_pis(self.X_val)
            clus_assign = self.model.clus_assign(self.X_val)

            # Define K
            K = self.model.K

            # Compute "hard" cluster assignment numbers
            hard_cluster_num = np.zeros(shape=K)
            for clus_id in range(self.K):
                hard_cluster_num[clus_id] = np.sum(clus_assign == clus_id)

            # Compute average cluster distribution
            avg_cluster_dist = np.mean(clus_pred, axis=0)

            # Print Information
            print(f"End of Epoch {epoch:d} - hard_cluster_info {hard_cluster_num} and avg dist{avg_cluster_dist}")


class SupervisedTargetMetrics(cbck.Callback):
    """
    Callback method to display supervised target metrics: Normalised Mutual Information, Adjusted Rand Score and
    Purity Score

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, *args, **kwargs):
        if epoch % self.interval == 0:
            # Compute y_pred, y_true in categorical format.
            model_output = (self.model(self.X_val)).numpy()
            class_pred = np.argmax(model_output, axis=-1)
            class_true = np.argmax(self.y_val, axis=-1).reshape(-1)

            # Target metrics
            nmi = normalized_mutual_info_score(labels_true=class_true, labels_pred=class_pred)
            ars = adjusted_rand_score(labels_true=class_true, labels_pred=class_pred)

            print("End of Epoch {:d} - NMI {:.2f} , ARS {:.2f}".format(epoch, nmi, ars))


class UnsupervisedTargetMetrics(cbck.Callback):
    """
    Callback method to display unsupervised target metrics: Davies-Bouldin Score, Calinski-Harabasz Score,
    Silhouette Score

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, *args, **kwargs):
        if epoch % self.interval == 0:
            # Compute predictions and latent representations
            latent_reps = self.model.Encoder(self.X_val)
            pis_pred = (self.model.Identifier(latent_reps)).numpy()

            # Convert to categorical
            clus_pred = np.argmax(pis_pred, axis=-1)

            # Reshape input data and allow feature comparison
            X_val_2d = np.reshape(self.X_val, (self.X_val.shape[0], -1))

            # Compute metrics
            dbs = davies_bouldin_score(X_val_2d, labels=clus_pred)
            dbs_l = davies_bouldin_score(latent_reps, labels=clus_pred)
            chs = calinski_harabasz_score(X_val_2d, labels=clus_pred)
            chs_l = calinski_harabasz_score(latent_reps, labels=clus_pred)
            sil = silhouette_score(X=X_val_2d, labels=clus_pred, random_state=self.model.seed)
            sil_l = silhouette_score(X=latent_reps, labels=clus_pred, random_state=self.model.seed)

            print(f"""End of Epoch {epoch:d} (score, latent score): 
                        DBS {dbs:.2f}, {dbs_l:.2f}   
                        CHS {chs:.2f}, {chs_l:.2f}  
                        SIL {sil:.2f}, {sil_l:.2f}""")


def cbck_list(summary_name: str, interval: int = 5, validation_data: tuple = ()):
    """
    Shorthand for callbacks above.

    Params:
    - summary_name: str containing shorthands for different callbacks.
    - interval: int interval to print information on.
    """
    extra_callback_list = []

    if "auc" in summary_name.lower() or "roc" in summary_name.lower():
        extra_callback_list.append(AUROC(interval=interval, validation_data=validation_data))

    if "clus_sep" in summary_name.lower() or "clus_phen" in summary_name.lower():
        extra_callback_list.append(CEClusSeparation(interval=interval, validation_data=validation_data))

    if "cm" in summary_name.lower() or "conf_matrix" in summary_name.lower():
        extra_callback_list.append(ConfusionMatrix(interval=interval, validation_data=validation_data))

    if "clus_info" in summary_name.lower():
        extra_callback_list.append(PrintClusterInfo(interval=interval, validation_data=validation_data))

    if "sup_scores" in summary_name.lower():
        extra_callback_list.append(SupervisedTargetMetrics(interval=interval, validation_data=validation_data))

    if "unsup_scores" in summary_name.lower():
        extra_callback_list.append(UnsupervisedTargetMetrics(interval=interval, validation_data=validation_data))

    return extra_callback_list


def get_callbacks(validation_data, data_name: str, track_loss: str, interval: int = 5, other_cbcks: str = "",
                  early_stop: bool = True, lr_scheduler: bool = True, tensorboard: bool = True,
                  min_delta: float = 0.0001, patience_epochs: int = 200):
    """
    Generate complete list of callbacks given input configuration.

    Params:
        - validation_data: tuple (X, y) of validation data.
        - data_name: str, data name on which the model is running
        - track_loss: str, name of main.py loss to keep track of.
        - interval: int, interval to print information on.
        - other_cbcks: str, list of other callbacks to consider (default = "", which selects None).
        - early_stop: whether to stop training early in case of no progress. (default = True)
        - lr_scheduler: dynamically update learning rate. (default = True)
        - tensorboard: write tensorboard friendly logs which can then be visualised. (default = True)
        - min_delta: if early stopping, the interval on which to check improvement or not.
        - patience_epochs: how many epochs to wait until checking for improvements.
        """

    # Initialise empty
    callbacks = []

    # Handle saving paths and folders
    logs_dir = LOGS_DIR
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Save Folder is first run that has not been previously computed
    run_num = 1
    while os.path.exists(logs_dir + f"{data_name}/run{run_num}/"):
        run_num += 1

    # Save as new run
    save_fd = logs_dir + f"{data_name}/run{run_num}/"
    assert not os.path.exists(save_fd)

    os.makedirs(save_fd)
    os.makedirs(save_fd + "logs/")
    os.makedirs(save_fd + "training/")

    # ------------------ Start Loading callbacks ---------------------------

    # Load custom callbacks first
    callbacks.extend(cbck_list(other_cbcks, interval, validation_data=validation_data))

    # Model Weight saving callback
    checkpoint = cbck.ModelCheckpoint(filepath=save_fd + "models/checkpoints/epoch-{epoch}", save_best_only=True,
                                      monitor=track_loss, save_freq="epoch")
    callbacks.append(checkpoint)

    # Logging Loss values)
    csv_logger = cbck.CSVLogger(filename=save_fd + "training/loss_tracker", separator=",", append=False)
    callbacks.append(csv_logger)

    # Check if Early stoppage is added
    if early_stop is True:
        callbacks.append(cbck.EarlyStopping(monitor='val_' + track_loss, mode="min", restore_best_weights=True,
                                            min_delta=min_delta, patience=patience_epochs))

    # Check if LR Scheduling is in place
    if lr_scheduler is True:
        callbacks.append(cbck.ReduceLROnPlateau(monitor='val_' + track_loss, mode='min', cooldown=15,
                                                min_lr=0.00001, factor=0.25))

    # Check if Tensorboard is active
    if tensorboard is True:
        callbacks.append(cbck.TensorBoard(log_dir=save_fd + "logs/", histogram_freq=1))

    return callbacks, run_num
