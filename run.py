# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
import wfdb
import glob
import shap
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# Initialize the models
models = {
    'Logistic Regression': LogisticRegression(),
    # 'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

def extract_pqrst_features(ecg_signal, sample_rate):
    cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=sample_rate)
    _, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=sample_rate)

    # Delineate the ECG signal
    _, waves_peak = nk.ecg_delineate(cleaned_ecg, rpeaks, sampling_rate=sample_rate, method="peak")

    rp_amp_idx = [i for i in range(len(waves_peak['ECG_P_Peaks'])) if
                  not math.isnan(rpeaks['ECG_R_Peaks'][i]) and not math.isnan(waves_peak['ECG_P_Peaks'][i])]
    rq_amp_idx = [i for i in range(len(waves_peak['ECG_Q_Peaks'])) if
                  not math.isnan(rpeaks['ECG_R_Peaks'][i]) and not math.isnan(waves_peak['ECG_Q_Peaks'][i])]
    rs_amp_idx = [i for i in range(len(waves_peak['ECG_S_Peaks'])) if
                  not math.isnan(rpeaks['ECG_R_Peaks'][i]) and not math.isnan(waves_peak['ECG_S_Peaks'][i])]
    rt_amp_idx = [i for i in range(len(waves_peak['ECG_T_Peaks'])) if
                  not math.isnan(rpeaks['ECG_R_Peaks'][i]) and not math.isnan(waves_peak['ECG_T_Peaks'][i])]

    amp_p = [cleaned_ecg[waves_peak['ECG_P_Peaks'][i]] for i in rp_amp_idx]
    amp_rp = [cleaned_ecg[rpeaks['ECG_R_Peaks'][i]] for i in rp_amp_idx]

    amp_q = [cleaned_ecg[waves_peak['ECG_Q_Peaks'][i]] for i in rq_amp_idx]
    amp_rq = [cleaned_ecg[rpeaks['ECG_R_Peaks'][i]] for i in rq_amp_idx]

    amp_s = [cleaned_ecg[waves_peak['ECG_S_Peaks'][i]] for i in rs_amp_idx]
    amp_rs = [cleaned_ecg[rpeaks['ECG_R_Peaks'][i]] for i in rs_amp_idx]

    amp_t = [cleaned_ecg[waves_peak['ECG_T_Peaks'][i]] for i in rt_amp_idx]
    amp_rt = [cleaned_ecg[rpeaks['ECG_R_Peaks'][i]] for i in rt_amp_idx]

    rp_amp = np.array([a - b for a, b in zip(amp_rp, amp_p)])
    rq_amp = np.array([a - b for a, b in zip(amp_rq, amp_q)])
    rs_amp = np.array([a - b for a, b in zip(amp_rs, amp_s)])
    rt_amp = np.array([a - b for a, b in zip(amp_rt, amp_t)])

    pq_dis = np.array([a - b for a, b in zip(waves_peak['ECG_Q_Peaks'], waves_peak['ECG_P_Peaks'])])
    qr_dis = np.array([a - b for a, b in zip(rpeaks['ECG_R_Peaks'], waves_peak['ECG_Q_Peaks'])])
    rs_dis = np.array([a - b for a, b in zip(waves_peak['ECG_S_Peaks'], rpeaks['ECG_R_Peaks'])])
    st_dis = np.array([a - b for a, b in zip(waves_peak['ECG_T_Peaks'], waves_peak['ECG_S_Peaks'])])

    rp_avg = np.mean(rp_amp[~np.isnan(rp_amp)])
    rq_avg = np.mean(rq_amp[~np.isnan(rq_amp)])
    rs_avg = np.mean(rs_amp[~np.isnan(rs_amp)])
    rt_avg = np.mean(rt_amp[~np.isnan(rt_amp)])

    pq_avg = np.mean(pq_dis[~np.isnan(pq_dis)])
    qr_avg = np.mean(qr_dis[~np.isnan(qr_dis)])
    rs_avg = np.mean(rs_dis[~np.isnan(rs_dis)])
    st_avg = np.mean(st_dis[~np.isnan(st_dis)])

    return [pq_avg, qr_avg, rs_avg, st_avg, rp_avg, rq_avg, rs_avg, rt_avg]


def main():
    # Replace 'sampledata' with the path to your .dat file (without the .dat extension)
    dat_files = glob.glob('/Users/ziyuwang/Desktop/research/biosignal_anonym/mit-bih-arrhythmia-database-1.0.0/*.dat')

    # Set sample-rate
    sample_rate = 360

    train_x_lst = []
    train_y_lst = []
    test_x_lst = []
    test_y_lst = []

    l30 = 0
    g30 = 0
    count = 0

    for j, file in enumerate(dat_files):
        print(file)

        signal, fields = wfdb.rdsamp(file[:-4])
        ecg_signal = signal[:, 0]  # Assuming the ECG signal is the first column
        gender = fields['comments'][0].split(' ')[1]
        age = int(fields['comments'][0].split(' ')[0])

        if (gender == 'M'):
            label = 0
        else:
            label = 1

        """if (age >= 35 and age <= 80):
            continue

        count = count+1

        if (age < 35):
            label=0#l30=l30+1
        elif (age > 80):
            label=1#g30=g30+1"""

        if (j <= 40):
            # Create train samples
            for i in range(30):
                start = i * (60 * sample_rate)
                end = (i + 1) * (60 * sample_rate)

                pqrst_features = extract_pqrst_features(ecg_signal[start:end], sample_rate)

                contains_nan = np.isnan(np.array(pqrst_features)).any()

                if (~contains_nan):
                    train_x_lst.append(pqrst_features)
                    train_y_lst.append(label)
        else:
            # Create test samples
            for i in range(30):
                start = i * (60 * sample_rate)
                end = (i + 1) * (60 * sample_rate)

                pqrst_features = extract_pqrst_features(ecg_signal[start:end], sample_rate)

                contains_nan = np.isnan(np.array(pqrst_features)).any()

                if (~contains_nan):
                    test_x_lst.append(pqrst_features)
                    test_y_lst.append(label)

    train_x_np = np.array(train_x_lst)
    train_y_np = np.array(train_y_lst)
    test_x_np = np.array(test_x_lst)
    test_y_np = np.array(test_y_lst)

    # Assuming train_x_np, train_y_np, test_x_np, and test_y_np are already defined arrays.

    # Shuffle training data
    train_x_np, train_y_np = shuffle(train_x_np, train_y_np, random_state=42)

    # Shuffle testing data
    test_x_np, test_y_np = shuffle(test_x_np, test_y_np, random_state=42)

    # Assuming your numpy arrays are defined as train_x_np, test_x_np, train_y_np, and test_y_np

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform it
    train_x_np = scaler.fit_transform(train_x_np)

    # Transform the test data with the same scaler
    test_x_np = scaler.transform(test_x_np)

    # Now train_x_np_scaled and test_x_np_scaled are the normalized versions of your datasets
    # Train and evaluate the models
    for name, model in models.items():
        model.fit(train_x_np, train_y_np)
        predictions = model.predict(test_x_np)
        accuracy = accuracy_score(test_y_np, predictions)
        confusion = confusion_matrix(test_y_np, predictions)
        print(f"{name}:")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(confusion)
        print("----\n")

        # Initialize a SHAP explainer object. The type of explainer depends on your model.
        # For tree-based models (like RandomForest, XGBoost, LightGBM), you can use TreeExplainer
        explainer = shap.Explainer(model, test_x_np)

        # Compute SHAP values. This can be computationally intensive for large datasets.
        shap_values = explainer(test_x_np)

        # SHAP values give us the contribution of each feature to the prediction for each instance
        # Summary plot for overall impact of features across all test instances
        shap.summary_plot(shap_values, test_x_np, plot_type="bar")

        # Detailed summary plot to show the impact of the features on model output
        shap.summary_plot(shap_values, test_x_np)

    # Feature importance from Random Forest
    print("Feature Importances from Random Forest:")
    importances = models['Random Forest'].feature_importances_
    print(importances)


if __name__ == "__main__":
    main()

