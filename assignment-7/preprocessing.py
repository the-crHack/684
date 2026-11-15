"""Train-only clinical preprocessing and stateless metadata feature hashing."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import FeatureHasher

NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]

class Encoder:
    """Fit clinical transforms on original training records only; hash metadata statelessly."""
    def __init__(self, categorical, bins):
        self.categorical = list(categorical)
        self.numeric = list(NUMERIC)
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
        self.hasher = FeatureHasher(n_features=bins, input_type="string", alternate_sign=False)

    def arrays(self, records):
        return (np.array([[float(r[k]) for k in self.numeric] for r in records]),
                np.array([[r[k] for k in self.categorical] for r in records], dtype=object))

    def fit(self, records):
        numeric, categorical = self.arrays(records)
        self.scaler.fit(numeric)
        self.ohe.fit(categorical)
        self.feature_names = (self.numeric + self.ohe.get_feature_names_out(self.categorical).tolist()
                              + [f"routing_hash_{i}" for i in range(self.hasher.n_features)])
        return self

    def transform(self, records):
        numeric, categorical = self.arrays(records)
        hashed = self.hasher.transform([["routing_tag=" + r["routing_tag"]] for r in records])
        return sparse.hstack([sparse.csr_matrix(self.scaler.transform(numeric)),
                              self.ohe.transform(categorical), hashed], format="csr")
