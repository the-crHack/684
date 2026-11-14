"""Shared model / training / evaluation code (Keras 3 on the JAX backend)."""
import json
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
from jax_privacy import keras_api  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

assert keras.config.backend() == "jax", "set KERAS_BACKEND=jax before importing keras"

# Fixed compute budget: every configuration processes the same number of
# examples (batch_size * train_steps = EXAMPLES), about 20 passes over the
# 30,200 member patients.
EXAMPLES = 614_400
BATCH_SIZES = [64, 128, 256, 512, 1024]
LEARNING_RATES = [0.1, 0.5, 1.0, 2.0, 4.0]
EPSILONS = [0.1, 1, 10, 100, 1000]
DELTA = 1e-6          # ~ 1/(30 n); see report for why not 1e-5
CLIP_NORM = 1.0
HIDDEN = 256
NOISE_CACHE = "results/noise_multipliers.json"


def load_data(path="data/patients.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def steps_for(batch_size, examples=EXAMPLES):
    return examples // batch_size


def build_model(n_features, seed=0):
    """The Assignment 1 target: 2-layer MLP, 256 ReLU units, logits out."""
    keras.utils.set_random_seed(seed)
    return keras.Sequential([
        keras.Input((n_features,)),
        keras.layers.Dense(HIDDEN, activation="relu"),
        keras.layers.Dense(3),
    ])


def _compile(model, lr):
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def _epochs_split(train_steps):
    """Split train_steps into (epochs, steps_per_epoch) with exact product."""
    for e in (10, 8, 5, 4, 2, 1):
        if train_steps % e == 0:
            return e, train_steps // e
    return 1, train_steps


def train_nondp(data, batch_size, lr, train_steps=None, seed=0):
    train_steps = train_steps or steps_for(batch_size)
    model = build_model(data["X_train"].shape[1], seed)
    _compile(model, lr)
    epochs, spe = _epochs_split(train_steps)
    # Plain shuffled minibatch SGD; repeat() so steps_per_epoch can exceed n/B.
    ds = _repeating_batches(data["X_train"], data["y_train"], batch_size, seed, spe)
    model.fit(ds, epochs=epochs, verbose=0)
    return model


class _repeating_batches(keras.utils.PyDataset):
    """Shuffled minibatches without replacement, reshuffled after each pass
    over the data; a Keras "epoch" is just `length` steps."""

    def __init__(self, X, y, bs, seed, length):
        super().__init__()
        self.X, self.y, self.bs, self.length = X, y, bs, length
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(X))
        self.ptr = 0

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.ptr + self.bs > len(self.X):
            self.order, self.ptr = self.rng.permutation(len(self.X)), 0
        idx = self.order[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return self.X[idx], self.y[idx]


def _load_cache():
    if os.path.exists(NOISE_CACHE):
        with open(NOISE_CACHE) as f:
            return json.load(f)
    return {}


def noise_multiplier(eps, batch_size, train_steps, n, delta=DELTA):
    """Calibrated sigma via jax_privacy's PLD accountant (cached on disk)."""
    key = f"eps={eps}|B={batch_size}|T={train_steps}|n={n}|delta={delta}"
    cache = _load_cache()
    if key not in cache:
        cfg = keras_api.DPKerasConfig(
            epsilon=eps, delta=delta, clipping_norm=CLIP_NORM,
            batch_size=batch_size, gradient_accumulation_steps=1,
            train_steps=train_steps, train_size=n, poisson_sampling_in_fit=True,
        ).update_with_calibrated_noise_multiplier()
        cache = _load_cache()
        cache[key] = cfg.noise_multiplier
        os.makedirs(os.path.dirname(NOISE_CACHE), exist_ok=True)
        tmp = f"{NOISE_CACHE}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=1)
        os.replace(tmp, NOISE_CACHE)  # atomic: parallel shards never see a half file
    return cache[key]


def train_dp(data, batch_size, lr, eps, train_steps=None, clip=CLIP_NORM,
             delta=DELTA, seed=0, X=None, y=None):
    """DP-SGD via jax_privacy.keras_api.make_private with Poisson sampling."""
    X = data["X_train"] if X is None else X
    y = data["y_train"] if y is None else y
    n = len(X)
    train_steps = train_steps or steps_for(batch_size)
    sigma = noise_multiplier(eps, batch_size, train_steps, n, delta)
    params = keras_api.DPKerasConfig(
        epsilon=eps,
        delta=delta,
        clipping_norm=clip,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=train_steps,
        train_size=n,
        poisson_sampling_in_fit=True,
        noise_multiplier=sigma,
        seed=seed,
    )
    model = build_model(X.shape[1], seed)
    model = keras_api.make_private(model, params)
    _compile(model, lr)
    epochs, spe = _epochs_split(train_steps)
    model.fit(X, y, batch_size=batch_size, epochs=epochs, steps_per_epoch=spe,
              verbose=0)
    return model, sigma


@jax.jit
def _per_example_ce(logits, y):
    logp = jax.nn.log_softmax(logits)
    return -jax.numpy.take_along_axis(logp, y[:, None], 1)[:, 0], jax.nn.softmax(logits)


def predict_logits(model, X):
    return np.asarray(model.predict(X, batch_size=8192, verbose=0))


def per_example_loss(model, X, y):
    loss, _ = _per_example_ce(predict_logits(model, X), y)
    return np.asarray(loss)


def evaluate(model, X, y):
    logits = predict_logits(model, X)
    loss, prob = _per_example_ce(logits, y)
    loss, prob = np.asarray(loss), np.asarray(prob)
    if not np.all(np.isfinite(prob)):
        return {"acc": float("nan"), "auroc": float("nan"), "logloss": float("nan")}
    return {
        "acc": float((prob.argmax(1) == y).mean()),
        "auroc": float(roc_auc_score(y, prob, multi_class="ovr", average="macro")),
        "logloss": float(loss.mean()),
    }
