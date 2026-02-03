import numpy as np
import os, gc, h5py, random, torch
from datetime import datetime

def createSplit(X_l, X_r, test_size):
    # X_l = np.array(X_l)
    # X_r = np.array(X_r)
    num_samples = len(X_l)
    test_size = int(num_samples * test_size)
    random_idx = np.random.permutation(num_samples)
    X_train_left = []
    X_train_right = []
    X_val_left = []
    X_val_right = []

    for idx in random_idx[test_size:]:
        X_train_left.append(X_l[idx])
        X_train_right.append(X_r[idx])
    
    for idx in random_idx[:test_size]:
        X_val_left.append(X_l[idx])
        X_val_right.append(X_r[idx])

    return X_train_left, X_train_right, X_val_left, X_val_right

def createChunks(path, context=100, test_size=0.15, verbose=True):
    X_left = []
    X_right = []
    X = np.load(path, mmap_mode='r').tolist()
    
    if context > len(X):
        raise ValueError(f"Context value cannot exceed minimum length data. Retry with a value <={len(X)}.", flush=True)
    if context < 50 :
        raise ValueError(f"Context value should have minimum length = 2. Retry with a value >= 50.", flush=True)
    
    context_l = context - 1
    for i in range(len(X) - context + 1):
        X_left.append(X[i : i+context_l])
        X_right.append(X[i+context_l : i+context])
    if verbose:
        print(f"Complete!", flush=True)
    del X
    gc.collect()
    print(f"X_left : {len(X_left),len(X_left[0]),len(X_left[0][0])}\tX_right : {len(X_right),len(X_right[0]),len(X_right[0][0])}", flush=True)
    X_train_left, X_train_right, X_test_left, X_test_right = createSplit(X_left, X_right, test_size)
    return X_train_left, X_train_right, X_test_left, X_test_right

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hrs = int(elapsed_time / 3600)
    elapsed_mins = int((elapsed_time - elapsed_hrs * 3600) / 60)
    elapsed_secs = elapsed_time - (elapsed_mins * 60 + elapsed_hrs * 3600)
    return elapsed_hrs, elapsed_mins, elapsed_secs

def getModelName(dataset, type, mode):
    now = str(datetime.now())
    date, time = now.split()[0], now.split()[1]
    date = date.split('-')
    date.reverse()
    date = '-'.join(date)
    time = time.replace(':', '-')[:8]

    model_name = f"{type}___{mode}___{dataset}___{date}_{time}"
    return model_name

def load_data(path):
    with h5py.File(path, 'r') as f:
        data = f['data']
        return list(data)

def load_single_sample(path):
    with h5py.File(path, 'r') as f:
        x_l = list(f['x_l'])
        x_r = list(f['x_r'])
        return np.array(x_l), np.array(x_r)

def getCausalMatrix(n_dim=None, data='henon'):
    if data=='henon':
        assert n_dim is not None
        GC = np.zeros([n_dim, n_dim])
        for i in range(n_dim):
            GC[i,i] = 1
            if i!=0:
                GC[i,i-1] = 1
        return GC

    if data=='lorenz':
        assert n_dim is not None
        GC = np.zeros((n_dim, n_dim), dtype=int)
        for i in range(n_dim):
            GC[i, i] = 1
            GC[i, (i + 1) % n_dim] = 1
            GC[i, (i - 1) % n_dim] = 1
            GC[i, (i - 2) % n_dim] = 1
        return GC
    
    if data=='ecoli':
        GC = np.array(
            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 0., 1., 0., 1., 1., 1., 1., 0.],
             [0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,],
             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
             [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.]]
        )
        return GC
    
    if data=='yeast':
        GC = np.array(
            [[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
             [0., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
        )
        return GC

def set_deterministic(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print("torch.use_deterministic_algorithms(True) not available on this PyTorch or may raise for certain ops:", e)