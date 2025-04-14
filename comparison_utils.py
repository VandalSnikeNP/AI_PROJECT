from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from Individual_class import *
from utilities_func import *
from MLP_Custom_Class import *

def training(model, train_loader, lr=0.001, epochs=1000, patience=20, early_stopping=True, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    epochs_no_improve = 0
    training_losses = []

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)

        if early_stopping:
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
                break
    print(f'Ended at Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
    return training_losses

def validate(model, validation_loader, device='cpu'):
    model.to(device)
    model.eval()
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for data, targets in validation_loader:
            data = data.to(device)
            outputs = model(data)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    final_mse = mean_squared_error(all_targets, all_outputs)
    print(f'Validation MSE: {final_mse:.9f}')

def plot_training_loss(training_loss1, training_loss2, separetely=True):
    plt.figure(figsize=(10, 5))
    if separetely:
        plt.subplot(1, 2, 1)
        plt.plot(training_loss1, label='Training Loss 1')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs for Training 1')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(training_loss2, label='Training Loss 2')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs for Training 2')
        plt.legend()
    else:
        plt.plot(training_loss1, label='Training Loss Sigmoid')
        plt.plot(training_loss2, label='Training Loss Best Individual')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

    plt.show()

def plot_model_comparisons(X_tensor, Y_tensor, model, net_test, point_size=2):
    model.eval()
    net_test.network.eval()

    with torch.no_grad():
        predictions_model = model(X_tensor).detach().numpy()
        predictions_net_test = net_test.network(X_tensor).detach().numpy()

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, (1, 2))
    plt.scatter(X_tensor, predictions_model, label="Output Sigmoid", s=point_size)
    plt.scatter(X_tensor, Y_tensor, label="Output Dataset", s=point_size)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Comparison Sigmoid")
    plt.legend()

    plt.subplot(1, 4, (3, 4))
    plt.scatter(X_tensor, predictions_net_test, label="Output Best Individual", s=point_size)
    plt.scatter(X_tensor, Y_tensor, label="Output Dataset", s=point_size)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Comparison Best individual")
    plt.legend()

    plt.tight_layout()
    plt.show()

class MLP_sigmoid(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pars_W,dict):
        self.seed=dict['seed']
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = torch.sigmoid
        self.fc2 = nn.Linear(hidden_size, output_size)

        if pars_W is not None:
            self.set_weights(pars_W)
        else:
            self.init_weights()

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

    def set_weights(self, pars_W):
        with torch.no_grad():
            for param, p in zip(self.parameters(), pars_W):
                param.copy_(torch.tensor(p))

def load_best_net(string_file,dict):
    net_data, pars = load_data(string_file)
    best_net=min(net_data[-1], key=lambda x: x.score)
    best_gen = best_net.genome
    return Individual(best_gen,dict['input_size'],dict['hidden_size'], dict['output_size'],pars=pars)


