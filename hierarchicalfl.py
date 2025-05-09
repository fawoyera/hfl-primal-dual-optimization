import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Configuration
class Config:
    n_clients = 100  # Total number of clients
    n_edges = 10     # Number of edge servers
    clients_per_edge = n_clients // n_edges
    n_rounds = 100    # Total communication rounds
    local_epochs = 5 # Local epochs per round
    lr = 0.01        # Learning rate
    reg = 0.01       # L2 regularization parameter
    # Communication and time constraints
    comm_limit = 10 ** 4.5  # Maximum bits per round
    time_limit = 0.9  # Maximum time per round (seconds)
    # Dual step sizes
    eta_lambda = 0.01
    eta_mu = 0.01

# 2. Load and preprocess data
def load_dataset(dataset_name, binary_class=0):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'fashion':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create binary classification (one vs rest)
    binary_train_dataset = [(x, 1 if y == binary_class else 0) for x, y in train_dataset]
    binary_test_dataset = [(x, 1 if y == binary_class else 0) for x, y in test_dataset]
    
    return binary_train_dataset, binary_test_dataset

# 3. Distribute data among clients
def distribute_data(train_dataset, n_clients):
    data_per_client = len(train_dataset) // n_clients
    client_data = []
    
    for i in range(n_clients):
        start_idx = i * data_per_client
        end_idx = start_idx + data_per_client
        client_data.append(train_dataset[start_idx:end_idx])
    
    return client_data

# 4. Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x.view(x.size(0), -1)))

# 5. Compute communication cost
def compute_comm_cost(model):
    # Count number of parameters in model
    num_params = sum(p.numel() for p in model.parameters())
    # Assume 32 bits (4 bytes) per parameter
    return num_params * 4 * 8  # in bits

# 6. Compute round time (simulated)
def compute_round_time(model, n_clients, n_edges):
    # Simple simulation of computation + communication time
    # Depends on model size and network topology
    model_size = sum(p.numel() for p in model.parameters())
    comp_time = 0.01 * model_size / 1000  # Computation time
    comm_time = 0.005 * model_size / 1000 * (n_clients/n_edges + n_edges)  # Communication time
    return comp_time + comm_time

# 7. Main HFL algorithm with Primal-Dual updates
def train_hfl(config, train_data, test_data, input_dim):
    # Initialize model
    global_model = LogisticRegression(input_dim)
    
    # Distribute data to clients
    client_data = distribute_data(train_data, config.n_clients)
    
    # Edge server assignment
    edge_clients = [list(range(i*config.clients_per_edge, (i+1)*config.clients_per_edge)) 
                   for i in range(config.n_edges)]
    
    # Initialize dual variables
    lambda_comm = 0.0
    mu_time = 0.0
    
    # Tracking metrics
    train_losses = []
    test_losses = []
    test_accs = []
    comm_costs = []
    round_times = []
    dual_values = []
    
    # Convert test data to tensors
    test_inputs = torch.stack([x for x, _ in test_data])
    test_labels = torch.tensor([y for _, y in test_data]).float().unsqueeze(1)
    
    for round_idx in range(config.n_rounds):
        round_start_time = time.time()
        
        # 1. Client Updates
        client_models = []
        for client_idx in range(config.n_clients):
            # Clone global model
            client_model = LogisticRegression(input_dim)
            client_model.load_state_dict(global_model.state_dict())
            
            # Get client data
            client_inputs = torch.stack([x for x, _ in client_data[client_idx]])
            client_labels = torch.tensor([y for _, y in client_data[client_idx]]).float().unsqueeze(1)
            
            # Train locally
            optimizer = optim.SGD(client_model.parameters(), lr=config.lr, weight_decay=config.reg)
            criterion = nn.BCELoss()
            
            for _ in range(config.local_epochs):
                # Forward pass
                outputs = client_model(client_inputs)
                
                # Compute loss with regularization
                loss = criterion(outputs, client_labels)
                
                # Add primal-dual terms for constraints
                comm_cost = compute_comm_cost(client_model)
                round_time = compute_round_time(client_model, config.n_clients, config.n_edges)
                
                primal_dual_loss = loss + lambda_comm * (comm_cost - config.comm_limit) + \
                                  mu_time * (config.n_rounds * round_time - config.time_limit)
                
                # Backward and optimize
                optimizer.zero_grad()
                primal_dual_loss.backward()
                optimizer.step()
            
            client_models.append(client_model)
        
        # 2. Edge Aggregation
        edge_models = []
        for edge_idx in range(config.n_edges):
            edge_clients_list = edge_clients[edge_idx]
            edge_model = LogisticRegression(input_dim)
            edge_model.load_state_dict(global_model.state_dict())  # Initialize
            
            # Aggregate client models within this edge
            with torch.no_grad():
                for name, param in edge_model.named_parameters():
                    # Average the parameters
                    param.data = torch.mean(torch.stack([client_models[c].state_dict()[name].data 
                                                        for c in edge_clients_list]), dim=0)
            
            edge_models.append(edge_model)
        
        # 3. Central Aggregation
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                # Average the parameters from all edge models
                param.data = torch.mean(torch.stack([em.state_dict()[name].data for em in edge_models]), dim=0)
        
        # 4. Evaluate current model
        global_model.eval()
        with torch.no_grad():
            test_outputs = global_model(test_inputs)
            test_loss = nn.BCELoss()(test_outputs, test_labels)
            test_preds = (test_outputs > 0.5).float()
            test_acc = (test_preds == test_labels).float().mean()
        
        # 5. Dual Updates
        comm_cost = compute_comm_cost(global_model)
        round_time = compute_round_time(global_model, config.n_clients, config.n_edges)
        
        # Update dual variables
        lambda_comm = max(0.0, lambda_comm + config.eta_lambda * (comm_cost - config.comm_limit))
        mu_time = max(0.0, mu_time + config.eta_mu * (config.n_rounds * round_time - config.time_limit))
        
        # 6. Record metrics
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        test_accs.append(test_acc.item())
        comm_costs.append(comm_cost)
        round_times.append(round_time)
        dual_values.append((lambda_comm, mu_time))
        
        actual_round_time = time.time() - round_start_time
        
        if round_idx % 5 == 0:
            print(f"Round {round_idx}: Test Loss: {test_loss.item():.4f}, Test Acc: {test_acc.item():.4f}, "
                  f"Comm Cost: {comm_cost/1000:.2f} Kb, Round Time: {actual_round_time:.2f}s, "
                  f"λ: {lambda_comm:.4f}, μ: {mu_time:.4f}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'comm_costs': comm_costs,
        'round_times': round_times,
        'dual_values': dual_values,
        'final_model': global_model
    }

# 8. Standard FedAvg for comparison
def train_fedavg(config, train_data, test_data, input_dim):
    # Similar to HFL but with direct client-to-central aggregation
    global_model = LogisticRegression(input_dim)
    client_data = distribute_data(train_data, config.n_clients)
    
    # Tracking metrics
    test_losses = []
    test_accs = []
    comm_costs = []
    round_times = []
    
    # Convert test data to tensors
    test_inputs = torch.stack([x for x, _ in test_data])
    test_labels = torch.tensor([y for _, y in test_data]).float().unsqueeze(1)
    
    for round_idx in range(config.n_rounds):
        round_start_time = time.time()
        
        # 1. Client Updates
        client_models = []
        for client_idx in range(config.n_clients):
            # Clone global model
            client_model = LogisticRegression(input_dim)
            client_model.load_state_dict(global_model.state_dict())
            
            # Get client data
            client_inputs = torch.stack([x for x, _ in client_data[client_idx]])
            client_labels = torch.tensor([y for _, y in client_data[client_idx]]).float().unsqueeze(1)
            
            # Train locally
            optimizer = optim.SGD(client_model.parameters(), lr=config.lr, weight_decay=config.reg)
            criterion = nn.BCELoss()
            
            for _ in range(config.local_epochs):
                # Forward pass
                outputs = client_model(client_inputs)
                loss = criterion(outputs, client_labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            client_models.append(client_model)
        
        # 2. Central Aggregation (no edge servers)
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                # Average the parameters from all client models
                param.data = torch.mean(torch.stack([cm.state_dict()[name].data for cm in client_models]), dim=0)
        
        # 3. Evaluate current model
        global_model.eval()
        with torch.no_grad():
            test_outputs = global_model(test_inputs)
            test_loss = nn.BCELoss()(test_outputs, test_labels)
            test_preds = (test_outputs > 0.5).float()
            test_acc = (test_preds == test_labels).float().mean()
        
        # 4. Record metrics
        comm_cost = config.n_clients * compute_comm_cost(global_model)  # Direct client-to-central communication
        round_time = time.time() - round_start_time
        
        test_losses.append(test_loss.item())
        test_accs.append(test_acc.item())
        comm_costs.append(comm_cost)
        round_times.append(round_time)
        
        if round_idx % 5 == 0:
            print(f"FedAvg Round {round_idx}: Test Loss: {test_loss.item():.4f}, "
                  f"Test Acc: {test_acc.item():.4f}, Comm Cost: {comm_cost/1000:.2f} Kb")
    
    return {
        'test_losses': test_losses,
        'test_accs': test_accs,
        'comm_costs': comm_costs,
        'round_times': round_times,
        'final_model': global_model
    }

# 9. Plot Convexity of Primal Problem
def plot_primal_convexity(model, test_data, reg=0.01):
    # Generate points around the optimal model parameters
    weights = next(model.parameters()).data.flatten()[0].item()
    w_range = np.linspace(weights - 2.0, weights + 2.0, 100)
    
    # Get test data
    test_inputs = torch.stack([x for x, _ in test_data[:1000]])  # Use subset for efficiency
    test_labels = torch.tensor([y for _, y in test_data[:1000]]).float().unsqueeze(1)
    
    # Compute loss for different weight values
    losses = []
    for w in w_range:
        temp_model = LogisticRegression(test_inputs.shape[1])
        # Set all weights to w
        with torch.no_grad():
            # Reshape the weights to match the expected shape
            reshaped_weights = torch.tensor(w * np.ones((1, test_inputs.shape[1] * test_inputs.shape[2] * test_inputs.shape[3])), dtype=torch.float32)
            temp_model.linear.weight.data = reshaped_weights
            temp_model.linear.bias.data.fill_(0.0)  # Reset bias
            #temp_model.linear.weight.data = torch.tensor(w * np.ones((1, test_inputs.shape[1])), dtype=torch.float32)
            #temp_model.linear.bias.data.fill_(0.0)  # Reset bias
            # param in temp_model.parameters():
                #param.data = torch.ones_like(param.data) * w  # Keep the original shape
                #param.data = torch.tensor(w * np.ones(param.shape), dtype=torch.float32)  # Initialize with the correct shape
                #param.fill_(w)
        
        # Compute loss
        outputs = temp_model(test_inputs)
        loss = nn.BCELoss()(outputs, test_labels) + reg * w**2
        losses.append(loss.item())
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(w_range, losses)
    plt.scatter([weights], [losses[np.argmin(np.abs(w_range - weights))]], color='red', s=100, marker='*')
    plt.title('Convexity of Primal Problem (Loss vs Weight)')
    plt.xlabel('Weight Value')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# 10. Plot Convexity of Dual Problem
def plot_dual_convexity(lambda_values, mu_values, dual_function):
    # Create meshgrid for 3D surface plot
    lambda_range = np.linspace(0, max(lambda_values)*1.5, 50)
    mu_range = np.linspace(0, max(mu_values)*1.5, 50)
    Lambda, Mu = np.meshgrid(lambda_range, mu_range)
    
    # Compute dual function values
    Z = np.zeros_like(Lambda)
    for i in range(len(lambda_range)):
        for j in range(len(mu_range)):
            Z[j, i] = dual_function(lambda_range[i], mu_range[j])
    
    # Plot 3D surface
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Lambda, Mu, Z, cmap='viridis', alpha=0.8)
    
    # Mark the maximum point (remember dual is maximized)
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    max_lambda, max_mu = lambda_range[max_idx[1]], mu_range[max_idx[0]]
    max_val = Z[max_idx]
    ax.scatter([max_lambda], [max_mu], [max_val], color='red', s=100, marker='*')
    
    # Mark the actual dual variables
    for i, (l, m) in enumerate(zip(lambda_values, mu_values)):
        if i % 5 == 0:  # Plot every 5th point to avoid clutter
            dual_val = dual_function(l, m)
            ax.scatter([l], [m], [dual_val], color='blue', s=50)
    
    # Add colorbar and labels
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('λ (Communication)')
    ax.set_ylabel('μ (Time)')
    ax.set_zlabel('Dual Function Value')
    ax.set_title('Convexity of Dual Problem')
    plt.show()


# Function to plot the evolution of dual variables
def plot_dual_evolution(lambda_hist, mu_hist):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(lambda_hist, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Lambda (λ)', fontsize=12)
    plt.title('Communication Constraint Dual Variable')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(mu_hist, 'r-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Mu (μ)', fontsize=12)
    plt.title('Time Constraint Dual Variable')
    plt.grid(True)

    plt.tight_layout()
    plt.show()  # Display the plot


# 11. Main execution function
def run_experiments():
    config = Config()
    results = {}
    
    for dataset_name in ['mnist', 'fashion', 'cifar10']:
        print(f"\n\n--- Running experiments on {dataset_name.upper()} ---")
        
        # Load data
        train_data, test_data = load_dataset(dataset_name)
        
        # Get input dimensions
        if dataset_name == 'mnist' or dataset_name == 'fashion':
            input_dim = 28 * 28
        else:  # cifar10
            input_dim = 32 * 32 * 3
        
        # Train HFL
        print("Training HFL with Primal-Dual updates...")
        hfl_results = train_hfl(config, train_data, test_data, input_dim)
        
        # Train FedAvg
        print("Training FedAvg for comparison...")
        fedavg_results = train_fedavg(config, train_data, test_data, input_dim)
        
        # Store results
        results[dataset_name] = {
            'hfl': hfl_results,
            'fedavg': fedavg_results
        }
        
        # Plot convexity of primal problem
        plot_primal_convexity(hfl_results['final_model'], test_data)
        
        # Simulate dual function for plotting
        def dual_function(lambda_val, mu_val):
            # This is a simplified representation of the dual function
            # In reality, you would need to solve the inner minimization problem
            return -1.0 * (lambda_val**2 + mu_val**2) + lambda_val * config.comm_limit + mu_val * config.time_limit
        
        # Plot convexity of dual problem
        lambda_values = [d[0] for d in hfl_results['dual_values']]
        mu_values = [d[1] for d in hfl_results['dual_values']]
        plot_dual_convexity(lambda_values, mu_values, dual_function)
    
        # Plot dual variable evolution for HFL
        lambda_hist = [d[0] for d in hfl_results['dual_values']]
        mu_hist = [d[1] for d in hfl_results['dual_values']]
        plot_dual_evolution(lambda_hist, mu_hist)

    # Create comparison plots
    for dataset_name in results:
        hfl = results[dataset_name]['hfl']
        fedavg = results[dataset_name]['fedavg']
        
        # Plot test accuracy
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(range(1, config.n_rounds + 1), hfl['test_accs'], label='HFL')
        plt.plot(range(1, config.n_rounds + 1), fedavg['test_accs'], label='FedAvg')
        plt.xlabel('Rounds')
        plt.ylabel('Test Accuracy')
        plt.title(f'{dataset_name.upper()}: Test Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot test loss
        plt.subplot(2, 2, 2)
        plt.plot(range(1, config.n_rounds + 1), hfl['test_losses'], label='HFL')
        plt.plot(range(1, config.n_rounds + 1), fedavg['test_losses'], label='FedAvg')
        plt.xlabel('Rounds')
        plt.ylabel('Test Loss')
        plt.title(f'{dataset_name.upper()}: Test Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot communication cost
        plt.subplot(2, 2, 3)
        plt.plot(range(1, config.n_rounds + 1), hfl['comm_costs'], label='HFL')
        plt.plot(range(1, config.n_rounds + 1), fedavg['comm_costs'], label='FedAvg')
        plt.xlabel('Rounds')
        plt.ylabel('Communication Cost (bits)')
        plt.title(f'{dataset_name.upper()}: Communication Cost')
        plt.legend()
        plt.grid(True)
        
        # Plot round time
        plt.subplot(2, 2, 4)
        plt.plot(range(1, config.n_rounds + 1), hfl['round_times'], label='HFL')
        plt.plot(range(1, config.n_rounds + 1), fedavg['round_times'], label='FedAvg')
        plt.xlabel('Rounds')
        plt.ylabel('Round Time (seconds)')
        plt.title(f'{dataset_name.upper()}: Round Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return results

# Run the experiments
if __name__ == "__main__":
    results = run_experiments()
