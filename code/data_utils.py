import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------------------
"Global variables for specific dataset information loading."

MIMIC_PARSE_TIME_VARS = ["intime", "outtime", "chartmax"]
MIMIC_PARSE_TD_VARS = [
    "sampled_time_to_end(1H)", "time_to_end", "time_to_end_min", "time_to_end_max"]
MIMIC_VITALS = ["TEMP", "HR", "RR", "SPO2", "SBP", "DBP"]
MIMIC_STATIC = ["age", "gender", "ESI"]
MIMIC_OUTCOME_NAMES = ["De", "I", "W", "Di"]

# Identifiers for main ids.
MAIN_ID_LIST = ["subject_id", "hadm_id", "stay_id", "patient_id", "pat_id"]

# ----------------------------------------------------------------------------------------


def convert_to_timedelta(df: pd.DataFrame, *args) -> pd.DataFrame:
    """Convert all given cols of dataframe to timedelta."""
    output = df.copy()
    for arg in args:
        output[arg] = pd.to_timedelta(df.loc[:, arg])

    return output


class CustomDataset(Dataset):

    def __init__(self, data_name="MIMIC", target_window=12, feat_set='vit-sta', time_range=(0, 6), parameters=None):
        if parameters is None:
            self.data_name = data_name
            self.target_window = target_window
            self.feat_set = feat_set
            self.time_range = time_range
            self.id_col = None
            self.time_col = None
            self.needs_time_to_end_computation = False
            self.min = None
            self.max = None

            # Load & process data
            self.id_col, self.time_col, self.needs_time_to_end_computation = self.get_ids(
                self.data_name)
            self.x, self.y, self.mask, self.pat_time_ids, self.features, self.outcomes, self.x_subset, self.y_data = self.load_transform()
        else:
            self.x, self.y, self.mask, self.pat_time_ids, self.features, self.outcomes, self.x_subset, self.y_data, self.id_col, self.time_col, self.needs_time_to_end_computation, self.data_name, self.feat_set, self.time_range, self.target_window, self.min, self.max = parameters

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :, :]
        y = self.y[idx, :]
        mask = self.mask[idx, :, :]
        pat_time_ids = self.pat_time_ids[idx, :, :]
        features = self.features
        outcomes = self.outcomes
        x_subset = self.x_subset[idx, :]
        y_data = self.y_data[idx, :]
        id_col = self.id_col
        time_col = self.time_col
        needs_time_to_end_computation = self.needs_time_to_end_computation
        data_name = self.data_name
        feat_set = self.feat_set
        time_range = self.time_range
        target_window = self.target_window
        min = self.min
        max = self.max
        return x, y, mask, pat_time_ids, features, outcomes, x_subset, y_data, id_col, time_col, needs_time_to_end_computation, data_name, feat_set, time_range, target_window, min, max

    def get_subset(self, idx):
        return CustomDataset(parameters=self[idx])

    def load_transform(self):
        # Load data
        data = self._load(self.data_name, window=self.target_window)
        self.id_col, self.time_col, self.needs_time_to_end_computation = self.get_ids(
            self.data_name)
        # print(data[0].shape, '0')
        x_inter = self._add_time_to_end(data[0])
        # print(x_inter.shape, '1')
        x_inter = self._truncate(x_inter)
        # print(x_inter.shape, '2')
        self._check_time_conversion(x_inter)

        # print(x_inter.shape, '3')
        x_subset, features = self._subset_to_features(x_inter)

        # print(x_inter.shape, '4')
        x_inter, pat_time_ids = self._convert_to_3d_arr(x_subset)
        x_subset = x_subset.to_numpy().astype(np.float32)

        # print(x_inter.shape, '5')
        x_inter = self._normalize(x_inter)

        # print(x_inter.shape, '6')
        x_out, mask = self._impute(x_inter)
        # print(x_out.shape, '7')

        outcomes = MIMIC_OUTCOME_NAMES
        y_data = data[1][outcomes]
        y_out = y_data.to_numpy().astype("float32")
        y_data = y_data.to_numpy().astype("float32")

        self._check_input_format(x_out, y_out)

        return x_out, y_out, mask, pat_time_ids, features, outcomes, x_subset, y_data

    def _load(self, data_name, window=4):

        data_fd = './data/MIMIC/processed/'
        # for Kaggle:
        # data_fd = f"/kaggle/input/mimic-processed/"
        try:
            os.path.exists(data_fd)
        except AssertionError:
            print(data_fd)


        X = pd.read_csv(data_fd + "vitals_process.csv",
                            parse_dates=MIMIC_PARSE_TIME_VARS, header=0, index_col=0)
        y = pd.read_csv(data_fd + f"outcomes_{window}h_process.csv", index_col=0)
        # for Kaggle:
        # X = pd.read_csv("vitals_process.csv", parse_dates=MIMIC_PARSE_TIME_VARS, header=0, index_col=0)
        # y = pd.read_csv(f"outcomes_{window}h_process.csv", index_col=0)

        X = convert_to_timedelta(X, *MIMIC_PARSE_TD_VARS)
        return X, y

    def get_ids(self, data_name):
        id_col, time_col, needs_time_to_end = "hadm_id", "sampled_time_to_end(1H)", False

        return id_col, time_col, needs_time_to_end

    def _impute(self, X):
        s1 = self._numpy_forward_fill(X)
        s2 = self._numpy_backward_fill(s1)
        s3 = self._median_fill(s2)
        mask = np.isnan(X)
        return s3, mask

    def _convert_datetime_to_hour(self, series):
        return series.dt.total_seconds() / 3600

    def _get_features(self, key, data_name="MIMIC"):
        if isinstance(key, list):
            return key

        elif isinstance(key, str):
            vitals = MIMIC_VITALS
            static = MIMIC_STATIC
            vars1, vars2 = None, None

            features = set([])
            if "vit" in key.lower():
                features.update(vitals)

            if "vars1" in key.lower():
                features.update(vars1)

            if "vars2" in key.lower():
                features.update(vars2)

            if "lab" in key.lower():
                features.update(vars1)
                features.update(vars2)

            if "sta" in key.lower():
                features.update(static)

            if "all" in key.lower():
                features = self._get_features("vit-lab-sta", data_name)

            sorted_features = sorted(features)
            print(
                f"\n{data_name} data has been subsettted to the following features: \n {sorted_features}.")

            return sorted_features


    def _numpy_forward_fill(self, array):
        arr_mask = np.isnan(array)
        arr_out = np.copy(array)
        arr_inter = np.where(~ arr_mask, np.arange(
            arr_mask.shape[1]).reshape(1, -1, 1), 0)
        np.maximum.accumulate(arr_inter, axis=1,
                              out=arr_inter)
        arr_out = arr_out[np.arange(arr_out.shape[0])[:, None, None],
                          arr_inter,
                          np.arange(arr_out.shape[-1])[None, None, :]]

        return arr_out

    def _numpy_backward_fill(self, array):
        arr_mask = np.isnan(array)
        arr_out = np.copy(array)

        arr_inter = np.where(~ arr_mask, np.arange(
            arr_mask.shape[1]).reshape(1, -1, 1), arr_mask.shape[1] - 1)
        arr_inter = np.minimum.accumulate(
            arr_inter[:, ::-1], axis=1)[:, ::-1]
        arr_out = arr_out[np.arange(arr_out.shape[0])[:, None, None],
                          arr_inter,
                          np.arange(arr_out.shape[-1])[None, None, :]]

        return arr_out

    def _median_fill(self, array):
        arr_mask = np.isnan(array)
        arr_out = np.copy(array)
        array_med = np.nanmedian(np.nanmedian(
            array, axis=0, keepdims=True), axis=1, keepdims=True)
        arr_out = np.where(arr_mask, array_med, arr_out)

        return arr_out


    def _check_input_format(self, X, y):
        try:
            assert X.shape[0] == y.shape[0]
            assert len(X.shape) == 3
            assert len(y.shape) == 2
            assert np.sum(np.isnan(X)) + np.sum(np.isnan(y)) == 0
            assert np.all(np.sum(y, axis=1) == 1)

        except Exception as e:
            print(e)
            raise AssertionError("Input format error.")

    def _add_time_to_end(self, X):
        x_inter = X.copy(deep=True)
        if self.needs_time_to_end_computation:
            times = X.groupby(self.id_col).apply(
                lambda x: x.loc[:, self.time_col].max() - x.loc[:, self.time_col])
            x_inter["time_to_end"] = self._convert_datetime_to_hour(
                times).values

        else:
            x_inter["time_to_end"] = x_inter[self.time_col].values
            x_inter["time_to_end"] = self._convert_datetime_to_hour(
                x_inter.loc[:, "time_to_end"])

        self.time_col = "time_to_end"
        x_out = x_inter.sort_values(
            by=[self.id_col, "time_to_end"], ascending=[True, False])

        return x_out

    def _truncate(self, X):
        try:
            min_time, max_time = self.time_range
            # print(self.time_range)
            return X[X['time_to_end'].between(min_time, max_time, inclusive="left")]

        except Exception:
            raise ValueError(f"Could not truncate.")

    def _check_time_conversion(self, X):
        min_time, max_time = self.time_range

        assert X[self.id_col].is_monotonic_increasing is True
        assert X.groupby(self.id_col).apply(
            lambda x: x["time_to_end"].is_monotonic_decreasing).all() == True
        assert X["time_to_end"].between(
            min_time, max_time, inclusive='left').all() == True

    def _subset_to_features(self, X):
        features = [self.id_col, "time_to_end"] + \
            self._get_features(self.feat_set, self.data_name)

        return X[features], features

    def _convert_to_3d_arr(self, X):
        max_time = X.groupby(self.id_col).count()["time_to_end"].max()
        num_ids = X[self.id_col].nunique()
        feats = [col for col in X.columns if col not in [
            self.id_col, "time_to_end"]]
        id_list = X[self.id_col].unique()

        array_out = np.empty(shape=(num_ids, max_time, len(feats)))
        array_out[:] = np.nan
        array_id_times = np.empty(shape=(num_ids, max_time, 2))
        array_id_times[:, :, 0] = np.repeat(np.expand_dims(
            id_list, axis=-1), repeats=max_time, axis=-1)

        for id_ in tqdm(id_list):
            index_ = np.where(id_list == id_)[0]
            x_id = X[X[self.id_col] == id_]

            x_id_copy = x_id.copy()
            x_id_copy["time_to_end"] = - x_id["time_to_end"].diff().values

            array_out[index_, :x_id_copy.shape[0], :] = x_id_copy[feats].values
            array_id_times[index_, :x_id_copy.shape[0],
                           1] = x_id["time_to_end"].values

        return array_out.astype("float32"), array_id_times.astype("float32")

    def _normalize(self, X):
        self.min = np.nanmin(X, axis=0, keepdims=True)
        self.max = np.nanmax(X, axis=0, keepdims=True)
        return np.divide(X - self.min, self.max - self.min)


# Custom Dataloader
def collate_fn(data):
    x, y, mask, pat_time_ids, features, outcomes, x_subset, y_data, id_col, time_col, needs_time_to_end_computation, data_name, feat_set, time_range, target_window, min, max = zip(
        *data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor(np.array(x))
    y = torch.tensor(np.array(y))
    x = x.to(device)
    y = y.to(device)

    return x, y


def load_data(train_dataset, val_dataset, test_dataset):

    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
