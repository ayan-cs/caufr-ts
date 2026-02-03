import torch, os, json, sys, time, gc
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from itertools import product

from utils import load_data, getModelName, epoch_time
from model import CausalForecaster

def train_epoch(model, train_dl, val_dl, optimizer, beta_kl, lam):
    total_loss = 0
    model.train()
    acc_steps = 1
    for idx, (X, y) in enumerate(train_dl):
        X = X.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        train_mse, train_kl = model(X, y)
        train_lasso = model.get_group_lasso_penalty()
        loss = train_mse + beta_kl * train_kl + lam * train_lasso
        loss /= acc_steps
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    train_loss = total_loss / len(train_dl.dataset)
    torch.cuda.empty_cache()

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.cuda(), y.cuda()
            val_mse, val_kl = model(X, y)
            val_lasso = model.get_group_lasso_penalty()
            loss = val_mse + beta_kl * val_kl + lam * val_lasso
            total_loss += loss.item() * X.size(0)
    val_loss = total_loss / len(val_dl.dataset)
    torch.cuda.empty_cache()
    
    return train_loss, val_loss, train_mse, train_kl, train_lasso, val_mse, val_kl, val_lasso

def grid_search_trainer(dataset_name, param_grid, X_train_left1, X_train_right1, X_val_left1, X_val_right1, patience, step_size):
    parent = os.path.abspath('')
    threshold = 'adaptive' # [float (0.01), None, 'adaptive']

    dataset_name, data_artifact = dataset_name
    print(f"Using dataset: {data_artifact}", flush=True)
    if not os.path.exists(os.path.join(parent, f'artifacts_{dataset_name}')):
        os.mkdir(os.path.join(parent, f'artifacts_{dataset_name}'))
    model_name = getModelName(dataset=dataset_name, type='trf', mode=f'{threshold}_global')
    artifact_path = os.path.join(parent, f'artifacts_{dataset_name}', model_name)
    os.mkdir(os.path.join(artifact_path))

    input_size = len(X_train_left1[0][0]) # X_train_left.shape[-1]
    seq_len = len(X_train_left1[0]) + 1 # X_train_left.shape[-2] + 1

    param_combinations = list(product(
    param_grid['lr'],
    param_grid['batch_size'],
    param_grid['d_model'],
    param_grid['latent_size'],
    param_grid['n_head'],
    param_grid['num_layers'],
    param_grid['dropout'],
    param_grid['beta_kl'],
    param_grid['lam_lasso']
    ))

    # best_val_loss = np.inf # For global best model (runtime error)
    epochs = 10000
    combination = 0
    metadata_list = []
    start_gs = time.time()
    for lr, batch_size, d_model, latent_size, n_head, num_layers, dropout, beta_kl, lam in param_combinations:
        combination += 1
        sys.stdout = open(os.path.join(artifact_path, f'train_comb{combination}.log'), 'w')
        print(f"Using dataset: {data_artifact}", flush=True)
        print(f"\nTraining with combination {combination} ::\nInitial LR: {lr}\tBatch size: {batch_size}\td_model: {d_model}\tlatent_size: {latent_size}\tn_heads: {n_head}\tnum_layers: {num_layers}\tDropout: {dropout}\tbeta_kl: {beta_kl}\tlam_lasso: {lam}", flush=True)
        X_train_left = np.array(X_train_left1)
        X_train_right = np.array(X_train_right1)
        train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train_left), torch.FloatTensor(X_train_right)), batch_size=batch_size, shuffle=True)
        del X_train_left, X_train_right
        gc.collect()
        
        X_val_left = np.array(X_val_left1)
        X_val_right = np.array(X_val_right1)
        val_dl = DataLoader(TensorDataset(torch.FloatTensor(X_val_left), torch.FloatTensor(X_val_right)), batch_size=batch_size)
        del X_val_left, X_val_right
        gc.collect()

        model = CausalForecaster(
            input_size=input_size, latent_size=latent_size,
            d_model=d_model, nhead=n_head, num_layers=num_layers, dropout=dropout
        ).cuda()

        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=step_size)

        train_loss_list = {
            'loss':[], 'mse':[], 'kl':[], 'lasso':[]
        }
        val_loss_list = {
            'loss':[], 'mse':[], 'kl':[], 'lasso':[]
        }
        GC_raw_list = []
        GC_list = []
        threshold_list = []
        variable_usage_list = []
        best_epoch = 0
        best_val_loss = np.inf # For best model in each combo (runs perfectly)
        step_counter = 0
        start = time.time()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}\tLearning rate : {scheduler.get_last_lr()}\n", flush=True)
            start_ep = time.time()
            train_loss, val_loss, train_mse, train_kl, train_lasso, val_mse, val_kl, val_lasso = train_epoch(model, train_dl, val_dl, optimizer, beta_kl, lam)
            end_ep = time.time()
            train_loss_list['loss'].append(train_loss)
            val_loss_list['loss'].append(val_loss)
            train_loss_list['mse'].append(train_mse.item())
            train_loss_list['kl'].append(train_kl.item())
            train_loss_list['lasso'].append(train_lasso.item())
            val_loss_list['mse'].append(val_mse.item())
            val_loss_list['kl'].append(val_kl.item())
            val_loss_list['lasso'].append(val_lasso.item())

            GC_raw, GC_est, thresh = model.get_causal_matrix(threshold=threshold)
            threshold_list.append(thresh)
            variable_usage_list.append(100 * torch.mean(GC_est).item())

            print(f"Train loss : {train_loss:10.6f}\nValidation loss : {val_loss:10.6f}", flush=True)
            print(f"Variable usage (Threshold={thresh}): {variable_usage_list[-1]:.2f}%", flush=True)
            if (epoch+1)%5==0:
                print(GC_est, flush=True)
            _, mn, sc = epoch_time(start_ep, end_ep)
            print(f"Epoch execution time : {mn}min. {sc:.6f}sec.", flush=True)

            GC_raw_list.append(GC_raw.tolist())
            GC_list.append(GC_est.tolist())

            if val_loss < best_val_loss :
                best_val_loss = val_loss
                step_counter = 0
                best_epoch = epoch+1
                metadata = {
                    'combination' : combination,
                    'dataset' : data_artifact,
                    'n_dim' : input_size,
                    'seq_len' : seq_len,
                    'artifact' : model_name,
                    'model' : {
                        'batch_size' : batch_size,
                        'input_size' : input_size,
                        'd_model' : d_model,
                        'latent_size': latent_size,
                        'n_head' : n_head,
                        'num_layers' : num_layers,
                        'dropout' : dropout
                    },
                    'max_epochs' : epoch,
                    'initial_lr' : lr,
                    'earlystopper_patience' : patience,
                    'lr_step' : step_size,
                    'beta_kl' : beta_kl,
                    'lam_lasso': lam,
                    'GC_raw': GC_raw.tolist(),
                    'GC_est': GC_est.tolist()
                }
                torch.save(model.state_dict(), os.path.join(artifact_path, f'checkpoint_comb{combination}.pt'))
                print(f"Model recorded with Val loss : {val_loss}", flush=True)
                best_loss = val_loss
                best_epoch = epoch
            else:
                step_counter += 1
                # step_counter = 0

            scheduler.step(val_loss)
            
            if step_counter >= patience:
                print(f"Model not improving. Moving on to next combination ...", flush=True)
                break
            torch.cuda.empty_cache()
        
        end = time.time()
        h, m, s = epoch_time(start, end)
        metadata['final_epoch'] = epoch
        metadata['optimal_epoch'] = best_epoch
        metadata['best_val_loss'] = best_loss
        metadata['training_time'] = {'hr' : h, 'mins' : m, 'sec' : s}
        metadata['avg_epoch_sec'] = (end - start)/(epoch+1)
        metadata['train_loss_list'] = train_loss_list
        metadata['val_loss_list'] = val_loss_list
        metadata['gc_raw_list'] = GC_raw_list
        metadata['gc_est_list'] = GC_list
        metadata['threshold_list'] = threshold_list
        metadata['variable_usage_list'] = variable_usage_list
        print(f"Total training time : {h}hrs. {m}mins. {s}sec.", flush=True)
        print("\n"+"#"*100+"\n"+"#"*100+"\n"+"#"*100+"\n", flush=True)
        sys.stdout = sys.__stdout__
        torch.cuda.empty_cache()
        metadata_list.append(metadata)
        with open(os.path.join(artifact_path, f'metadata_comb{combination}.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
    
    end_gs = time.time()
    h, m, s = epoch_time(start_gs, end_gs)
    print(f"Total Grid Search training time : {h}hrs. {m}mins. {s}sec.", flush=True)
    metadata_list.append({'grid_search_time' : {'hr' : h, 'mins' : m, 'sec' : s}})
    # sys.stdout = sys.__stdout__
    return metadata_list

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    parent = os.path.abspath('')

    ####################################################################
    ### Manually change this place for other datasets
    dataset_name = 'henon' # ['henon', 'lorenz', 'ecoli', 'yeast']
    dataset_artifact = 'henon_5000_10'
    context = 50
    datapath = os.path.join(parent, 'datasets', dataset_artifact)
    ####################################################################

    print(f"Loading dataset : {dataset_name} ...", flush=True)
    X_train_left = load_data(os.path.join(datapath, f'{dataset_name}___X_train_left_{context}.h5'))
    X_train_right = load_data(os.path.join(datapath, f'{dataset_name}___X_train_right_{context}.h5'))
    X_val_left = load_data(os.path.join(datapath, f'{dataset_name}___X_val_left_{context}.h5'))
    X_val_right = load_data(os.path.join(datapath, f'{dataset_name}___X_val_right_{context}.h5'))
    print(f"Dataset loaded.", flush=True)

    param_grid = {
    "lr" : [0.001],
    "batch_size" : [256],
    "d_model" : [32],
    "latent_size": [8],
    "n_head" : [4],
    "num_layers" : [1],
    "dropout" : [0],
    "beta_kl": [0.01],
    "lam_lasso": [0.1]
    }

    metadata_list = grid_search_trainer(
        dataset_name=(dataset_name, dataset_artifact),
        param_grid=param_grid,
        X_train_left1=X_train_left, X_train_right1=X_train_right, X_val_left1=X_val_left, X_val_right1=X_val_right,
        patience=100, step_size=20
    )

    with open(os.path.join(parent, f'artifacts_{dataset_name}', metadata_list[0]['artifact'], 'train_metadata_all.jsonl'), 'w') as f:
        for obj in metadata_list:
            f.write(json.dumps(obj)+'\n')