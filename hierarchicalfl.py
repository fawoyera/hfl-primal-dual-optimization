import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import matplotlib.pyplot as plt

# ====== Configurations ======
DATASETS = ['MNIST', 'FMNIST', 'CIFAR10']
NUM_CLIENTS = 30
NUM_EDGES = 3
CLIENTS_PER_EDGE = NUM_CLIENTS // NUM_EDGES
ROUNDS = 20
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LR = 0.05
TAU = 2  # local aggregation frequency
MODEL_BITS = 32  # bits per parameter
B_CLIENT_EDGE = 2e6  # 2 Mbps
B_EDGE_CLOUD = 5e6   # 5 Mbps
C_TOTAL_MAX = 1e6    # total communication budget (bits)
T_MAX = 50          # total time budget (seconds)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ====== Model: Logistic Regression (Convex) ======
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

def get_model(dataset):
    if dataset in ['MNIST', 'FMNIST']:
        input_dim = 28 * 28
        num_classes = 10
    elif dataset == 'CIFAR10':
        input_dim = 32 * 32 * 3
        num_classes = 10
    else:
        raise ValueError("Unknown dataset")
    return LogisticRegression(input_dim, num_classes)

# ====== Data Loading and Partitioning ======
def load_data(dataset):
    if dataset == 'MNIST':
        ds = datasets.MNIST
        tfm = transforms.Compose([transforms.ToTensor()])
    elif dataset == 'FMNIST':
        ds = datasets.FashionMNIST
        tfm = transforms.Compose([transforms.ToTensor()])
    elif dataset == 'CIFAR10':
        ds = datasets.CIFAR10
        tfm = transforms.Compose([transforms.ToTensor()])
    else:
        raise ValueError("Unknown dataset")
    train = ds('./data', train=True, download=True, transform=tfm)
    test = ds('./data', train=False, download=True, transform=tfm)
    return train, test

def partition_data(dataset, num_clients):
    train, test = load_data(dataset)
    data_per_client = len(train) // num_clients
    indices = np.random.permutation(len(train))
    client_indices = [indices[i*data_per_client:(i+1)*data_per_client] for i in range(num_clients)]
    return [Subset(train, idxs) for idxs in client_indices], test

# ====== Communication and Time Functions ======
def model_size_bits(model):
    return sum(p.numel() for p in model.parameters()) * MODEL_BITS

def comm_cost_per_round(M, K, S, tau):
    return (M * S) // tau + K * S

def round_time(tau, B, C_i, M, K, S, B_client_edge, B_edge_cloud):
    T_comp = max([(tau * B) / ci for ci in C_i])
    T_comm = (M * S) / B_client_edge + (K * S) / B_EDGE_CLOUD
    return T_comp + T_comm

# ====== Federated Training ======
def client_update(model, dataloader, epochs, lr):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def aggregate_models(models):
    avg_state = {}
    for k in models[0].keys():
        avg_state[k] = sum([m[k] for m in models]) / len(models)
    return avg_state

def evaluate(model, test_loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total, total_loss / total

# ====== HFL Primal-Dual Algorithm ======
def hfl_primal_dual_experiment(dataset):
    client_datasets, test_data = partition_data(dataset, NUM_CLIENTS)
    test_loader = DataLoader(test_data, batch_size=256)
    global_model = get_model(dataset).to(DEVICE)
    S = model_size_bits(global_model)
    C_i = np.random.uniform(100, 300, NUM_CLIENTS)  # samples/sec

    lambda_dual, mu_dual = 0.0, 0.0
    eta_lambda, eta_mu = 1e-7, 1e-4

    lambda_hist, mu_hist, acc_hist, comm_hist, time_hist, test_loss_hist, primal_norm_hist = [], [], [], [], [], [], []
    total_comm, total_time = 0, 0

    for rnd in range(ROUNDS):
        edge_clients = [list(range(i*CLIENTS_PER_EDGE, (i+1)*CLIENTS_PER_EDGE)) for i in range(NUM_EDGES)]
        client_models = []
        for edge in edge_clients:
            edge_model = get_model(dataset).to(DEVICE)
            edge_model.load_state_dict(global_model.state_dict())
            local_models = []
            for cid in edge:
                loader = DataLoader(client_datasets[cid], batch_size=BATCH_SIZE, shuffle=True)
                local_model = get_model(dataset).to(DEVICE)
                local_model.load_state_dict(edge_model.state_dict())
                local_state = client_update(local_model, loader, LOCAL_EPOCHS, LR)
                local_models.append(local_state)
            edge_state = aggregate_models(local_models)
            edge_model.load_state_dict(edge_state)
            client_models.append(edge_model.state_dict())
        global_state = aggregate_models(client_models)
        global_model.load_state_dict(global_state)

        C_round = comm_cost_per_round(NUM_CLIENTS, NUM_EDGES, S, TAU)
        T_round = round_time(TAU, BATCH_SIZE, C_i, NUM_CLIENTS, NUM_EDGES, S, B_CLIENT_EDGE, B_EDGE_CLOUD)
        C_budget = C_TOTAL_MAX / ROUNDS
        T_budget = T_MAX / ROUNDS

        lambda_dual = max(0, lambda_dual + eta_lambda * (C_round - C_budget))
        mu_dual = max(0, mu_dual + eta_mu * (T_round - T_budget))

        lambda_hist.append(lambda_dual)
        mu_hist.append(mu_dual)
        comm_hist.append(C_round)
        time_hist.append(T_round)
        total_comm += C_round
        total_time += T_round

        acc, test_loss = evaluate(global_model, test_loader)
        acc_hist.append(acc)
        test_loss_hist.append(test_loss)
        # Primal norm (for plotting)
        with torch.no_grad():
            w_vec = torch.cat([p.view(-1) for p in global_model.parameters()])
            primal_norm_hist.append(w_vec.norm().item())

        print(f"[HFL-{dataset}] Round {rnd+1}/{ROUNDS} | Acc: {acc:.4f} | Loss: {test_loss:.4f} | Lambda: {lambda_dual:.2e} | Mu: {mu_dual:.2e} | Comm: {C_round:.2e} | Time: {T_round:.2f}")

    return {
        'acc': acc_hist, 'loss': test_loss_hist, 'lambda': lambda_hist, 'mu': mu_hist,
        'comm': comm_hist, 'time': time_hist, 'primal_norm': primal_norm_hist,
        'model': global_model, 'test_loader': test_loader, 'lambda_star': lambda_dual, 'mu_star': mu_dual,
        'C_budget': C_budget, 'T_budget': T_budget, 'S': S, 'C_i': C_i
    }

# ====== FedAvg Baseline ======
def fedavg_experiment(dataset):
    client_datasets, test_data = partition_data(dataset, NUM_CLIENTS)
    test_loader = DataLoader(test_data, batch_size=256)
    global_model = get_model(dataset).to(DEVICE)
    S = model_size_bits(global_model)
    C_i = np.random.uniform(100, 300, NUM_CLIENTS)  # samples/sec

    acc_hist, comm_hist, time_hist, test_loss_hist, primal_norm_hist = [], [], [], [], []
    total_comm, total_time = 0, 0

    for rnd in range(ROUNDS):
        local_states = []
        for cid in range(NUM_CLIENTS):
            loader = DataLoader(client_datasets[cid], batch_size=BATCH_SIZE, shuffle=True)
            local_model = get_model(dataset).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            local_state = client_update(local_model, loader, LOCAL_EPOCHS, LR)
            local_states.append(local_state)
        avg_state = aggregate_models(local_states)
        global_model.load_state_dict(avg_state)

        # Communication cost: all clients to server, no hierarchy
        C_round = NUM_CLIENTS * S
        # Time: slowest client
        T_comp = max([(TAU * BATCH_SIZE) / ci for ci in C_i])
        T_comm = (NUM_CLIENTS * S) / B_CLIENT_EDGE
        T_round = T_comp + T_comm

        comm_hist.append(C_round)
        time_hist.append(T_round)
        total_comm += C_round
        total_time += T_round

        acc, test_loss = evaluate(global_model, test_loader)
        acc_hist.append(acc)
        test_loss_hist.append(test_loss)
        with torch.no_grad():
            w_vec = torch.cat([p.view(-1) for p in global_model.parameters()])
            primal_norm_hist.append(w_vec.norm().item())

        print(f"[FedAvg-{dataset}] Round {rnd+1}/{ROUNDS} | Acc: {acc:.4f} | Loss: {test_loss:.4f} | Comm: {C_round:.2e} | Time: {T_round:.2f}")

    return {
        'acc': acc_hist, 'loss': test_loss_hist, 'comm': comm_hist, 'time': time_hist, 'primal_norm': primal_norm_hist,
        'model': global_model, 'test_loader': test_loader
    }

# ====== Convexity and Strong Duality Plots ======
def plot_primal_convexity(global_model, test_loader, dataset):
    w0 = torch.cat([p.detach().flatten() for p in global_model.parameters()])
    direction = torch.randn_like(w0)
    direction = direction / torch.norm(direction)
    t_vals = np.linspace(-2.5, 3.5, 41)
    primal_vals = []
    for t in t_vals:
        offset = (w0 + t*direction).clone()
        idx = 0
        new_state = {}
        for name, param in global_model.named_parameters():
            numel = param.numel()
            new_state[name] = offset[idx:idx+numel].reshape(param.shape)
            idx += numel
        test_model = get_model(dataset).to(DEVICE)
        test_model.load_state_dict(new_state)
        test_model.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = test_model(x)
                total_loss += loss_fn(logits, y).item() * y.size(0)
                total += y.size(0)
        Fw = total_loss / total
        primal_vals.append(Fw)
    plt.figure()
    plt.plot(t_vals, primal_vals, label='Primal Objective $F(w)$')
    plt.xlabel('Line search parameter $t$')
    plt.ylabel('Objective value')
    plt.title('Convexity of Primal Objective')
    plt.legend()
    plt.grid()
    plt.show()

def plot_dual_convexity(global_model, dataset, S, C_i, test_loader):
    # Define grid for lambda and mu
    lambda_vals = np.linspace(0, 2e-5, 21)
    mu_vals = np.linspace(0, 2e-2, 21)
    dual_grid = np.zeros((len(lambda_vals), len(mu_vals)))

    # Evaluate dual function at each (lambda, mu) pair
    for i, lmbd in enumerate(lambda_vals):
        for j, mu in enumerate(mu_vals):
            # For each (lambda, mu), evaluate L(w, lambda, mu) at current w (approximation)
            model = get_model(dataset).to(DEVICE)
            model.load_state_dict(global_model.state_dict())
            loss_fn = torch.nn.CrossEntropyLoss()
            total_loss, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits = model(x)
                    total_loss += loss_fn(logits, y).item() * y.size(0)
                    total += y.size(0)
            Fw = total_loss / total
            # Constraints (simulate as constants for visualization)
            C_round = comm_cost_per_round(NUM_CLIENTS, NUM_EDGES, S, TAU)
            T_round = round_time(TAU, BATCH_SIZE, C_i, NUM_CLIENTS, NUM_EDGES, S, B_CLIENT_EDGE, B_EDGE_CLOUD)
            dual_grid[i, j] = Fw + lmbd * (C_round) + mu * (T_round)

    # 3D surface plot
    Lambda, Mu = np.meshgrid(mu_vals, lambda_vals)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Lambda, Mu, dual_grid, cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\lambda$')
    ax.set_zlabel(r'$g(\lambda, \mu)$')
    ax.set_title('Convexity of Dual Function $g(\lambda, \mu)$')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


# ====== Main Experiment Loop ======
for ds in DATASETS:
    print("\n" + "="*40)
    print(f"Running experiments for {ds}")
    print("="*40)
    hfl_res = hfl_primal_dual_experiment(ds)
    fedavg_res = fedavg_experiment(ds)

    # 1. Convexity of Primal and Dual
    plot_primal_convexity(hfl_res['model'], hfl_res['test_loader'], ds)
    plot_dual_convexity(hfl_res['model'], ds, hfl_res['S'], hfl_res['C_i'], hfl_res['test_loader'])

    # 2. Test loss, accuracy, C_round, T_round vs rounds (HFL vs FedAvg)
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    plt.plot(hfl_res['loss'], label='HFL')
    plt.plot(fedavg_res['loss'], label='FedAvg', linestyle='--')
    plt.title('Test Loss'); plt.xlabel('Round')
    plt.legend(); plt.grid()
    plt.subplot(2,2,2)
    plt.plot(hfl_res['acc'], label='HFL')
    plt.plot(fedavg_res['acc'], label='FedAvg', linestyle='--')
    plt.title('Test Accuracy'); plt.xlabel('Round')
    plt.legend(); plt.grid()
    plt.subplot(2,2,3)
    plt.plot(hfl_res['comm'], label='HFL')
    plt.plot(fedavg_res['comm'], label='FedAvg', linestyle='--')
    plt.title('C_round'); plt.xlabel('Round')
    plt.legend(); plt.grid()
    plt.subplot(2,2,4)
    plt.plot(hfl_res['time'], label='HFL')
    plt.plot(fedavg_res['time'], label='FedAvg', linestyle='--')
    plt.title('T_round'); plt.xlabel('Round')
    plt.legend(); plt.grid()
    plt.tight_layout(); plt.show()

    # 3. Evolution of primal and dual variables (HFL)
    plt.figure()
    plt.plot(hfl_res['primal_norm'], label='||w|| (primal)')
    plt.plot(hfl_res['lambda'], label='Lambda (dual)')
    plt.plot(hfl_res['mu'], label='Mu (dual)')
    plt.xlabel('Round'); plt.title('Evolution of Primal/Dual Variables (HFL)')
    plt.legend(); plt.grid(); plt.show()

    # 4. Final comparison summary
    print(f"\n[Summary: {ds}]")
    print(f"  HFL Final Accuracy:   {hfl_res['acc'][-1]:.4f}")
    print(f"  FedAvg Final Accuracy:{fedavg_res['acc'][-1]:.4f}")
    print(f"  HFL Total Comm:       {sum(hfl_res['comm']):.2e} bits")
    print(f"  FedAvg Total Comm:    {sum(fedavg_res['comm']):.2e} bits")
    print(f"  HFL Total Time:       {sum(hfl_res['time']):.2f} sec")
    print(f"  FedAvg Total Time:    {sum(fedavg_res['time']):.2f} sec")
