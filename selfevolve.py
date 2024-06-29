import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import json
import openai
import copy
import random


# .envファイルの内容を読み込見込む
load_dotenv()
# OpenAI APIキーの設定
openai.organization = os.environ["OPENAI_ORGANIZATION"]
openai.api_key=os.environ['OPENAI_API_KEY']

from openai import OpenAI
client = OpenAI(api_key = os.environ['OPENAI_API_KEY'])

# seed値の固定
def fix_all_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ランダムに0以外のfloat型の数をn個生成
def generate_non_zero_floats(n):
    data = []
    while len(data) < n:
        num = np.random.uniform(-10000, 10000)  # -10000から10000の範囲で生成
        if num != 0:
            data.append(num)
    return np.array(data)

# ニューラルネットワークの定義
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(1, 3)
        self.layer2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

    def get_weights(self):
        return {name: param.data.tolist() for name, param in self.named_parameters()}

    def set_weights(self, weights):
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(torch.tensor(weights[name]))

def get_gpt4_response(prompt):    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    return response.choices[0].message.content

# 学習結果をプロットする関数
def plot_result(loss_list,title,dir_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list[0], label='Baseline')
    plt.plot(loss_list[1], label='Self-Evolving')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.xlim(0, 100)
    if 'acc' in title.lower():
        plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(dir_path+f'{title}.png')
    plt.close()

def run_training(epoch_num,threshold,model,optimizer,criterion,X_train,y_train,X_val,y_val,X_test,y_test):

    # 学習曲線用のリスト
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    train_loss,train_accuracy,val_loss,val_accuracy = eval(model,threshold,criterion,X_train,y_train,X_val,y_val)
    best_train_accuracy = train_accuracy
    best_val_accuracy = val_accuracy
    best_model_wights = copy.deepcopy(model.state_dict())
    
    print(f'Inital state, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # 学習ループ
    for epoch in range(epoch_num):
        # 学習
        model.train()
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train.unsqueeze(1))
        train_predicted = (train_outputs > threshold).float()
        train_accuracy = (train_predicted.squeeze() == y_train).float().mean().item()
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # 検証
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.unsqueeze(1))
            val_predicted = (val_outputs > threshold).float()
            val_accuracy = (val_predicted.squeeze() == y_val).float().mean().item()
        
        # 損失と精度を記録
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    
        if val_accuracy > best_val_accuracy:
            best_model_wights = copy.deepcopy(model.state_dict())
            best_val_accuracy = val_accuracy

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epoch_num}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Best Val Accuracy: {best_val_accuracy:.4f}')


    model = MLP()
    model.load_state_dict(best_model_wights)
    model.eval()
    # モデルの評価（テストデータ）
    with torch.no_grad():
        test_outputs = model(X_test)
        test_predicted = (test_outputs > threshold).float()
        test_correct = (test_predicted.squeeze() == y_test).sum().item()
        test_accuracy = 100 * test_correct / len(y_test)
        print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def eval(model,threshold,criterion,X_train,y_train,X_val,y_val):
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train.unsqueeze(1)).item()
        train_predicted = (train_outputs > threshold).float()
        train_accuracy = (train_predicted.squeeze() == y_train).float().mean().item()

    # 検証データで推論
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()
        val_predicted = (val_outputs > threshold).float()
        val_accuracy = (val_predicted.squeeze() == y_val).float().mean().item()
    
    return train_loss,train_accuracy,val_loss,val_accuracy

def run_selfevolve(epoch_num,threshold,model,criterion,X_train,y_train,X_val,y_val,X_test,y_test):

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    train_loss,train_accuracy,val_loss,val_accuracy = eval(model,threshold,criterion,X_train,y_train,X_val,y_val)
    best_train_accuracy = train_accuracy
    best_val_accuracy = val_accuracy
    best_model_wights = copy.deepcopy(model.state_dict())
        
    weights = model.get_weights()
    prompt = f"{json.dumps(weights)}\n"
    prompt += f"This is the model weights. The train loss was {train_loss:.4f} and train accuracy was {train_accuracy:.4f}. Please optimize the weights to reduce training loss and improve accuracy, then output the model weights in JSON format. The magnitude of weight changes is up to you.\n"
    gpt4_output = get_gpt4_response(prompt)
    
    print(f'Inital state, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    for epoch in range(epoch_num):
        try:
            model.set_weights(json.loads(gpt4_output))
        except:
            pass
        
        train_loss,train_accuracy,val_loss,val_accuracy = eval(model,threshold,criterion,X_train,y_train,X_val,y_val)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epoch_num}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Best Val Accuracy: {best_val_accuracy:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        if train_accuracy > best_train_accuracy:
            prompt += f"Previous your output: {gpt4_output}\n"
            prompt += f"The train loss was {train_loss:.4f} and train accuracy was {train_accuracy:.4f}. Considering the previous inputs and your outputs, please optimize the weights to reduce the learning loss to 0 and improve accuracy to 1.0, then output the model weights in JSON format. The magnitude of the weight changes is up to you.\n"
            best_train_accuracy = train_accuracy
        
        if val_accuracy > best_val_accuracy:
            best_model_wights = copy.deepcopy(model.state_dict())
            best_val_accuracy = val_accuracy
        
        try:
            gpt4_output = get_gpt4_response(prompt)
        except:
            print('Prompt is too long.')
            break

    # モデルの評価（テストデータ）
    model = MLP()
    model.load_state_dict(best_model_wights)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_predicted = (test_outputs > threshold).float()
        test_correct = (test_predicted.squeeze() == y_test).sum().item()
        test_accuracy = 100 * test_correct / len(y_test)
        print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def run(epoch_num,threshold,data_num,dir_path):
    os.makedirs(dir_path, exist_ok=True)
    
    # データセットの作成
    X = generate_non_zero_floats(data_num)  # 1000000個のデータを生成
    y = (X > 0).astype(int)

    # データを学習用、検証用、テスト用に分割 (60% 学習, 20% 検証, 20% テスト)
    train_split = int(0.6 * len(X))
    val_split = int(0.8 * len(X))
    X_train, X_val, X_test = X[:train_split], X[train_split:val_split], X[val_split:]
    y_train, y_val, y_test = y[:train_split], y[train_split:val_split], y[val_split:]

    # PyTorchのテンソルに変換
    X_train = torch.FloatTensor(X_train).view(-1, 1)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val).view(-1, 1)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test).view(-1, 1)
    y_test = torch.FloatTensor(y_test)

    # モデルの初期化
    initial_model = MLP()
    model_training = copy.deepcopy(initial_model)
    model_selfevolve = copy.deepcopy(initial_model)

    # 損失関数と最適化アルゴリズムの定義
    criterion = nn.BCELoss()

    # Training model
    optimizer = optim.Adam(model_training.parameters())
    print('Training model...')
    train_losses1, val_losses1, train_accuracies1, val_accuracies1 = run_training(epoch_num,threshold,model_training,optimizer,criterion,X_train,y_train,X_val,y_val,X_test,y_test)
    
    print()
    
    # Self-Evolving model
    print('Self-evolving model...')
    train_losses2, val_losses2, train_accuracies2, val_accuracies2 = run_selfevolve(epoch_num,threshold,model_selfevolve,criterion,X_train,y_train,X_val,y_val,X_test,y_test)

    plot_result([train_losses1, train_losses2], 'Train_Loss', dir_path)
    plot_result([val_losses1, val_losses2], 'Val_Loss', dir_path)
    plot_result([train_accuracies1, train_accuracies2], 'Train_Acc', dir_path)
    plot_result([val_accuracies1, val_accuracies2], 'Val_Acc', dir_path)

if __name__ == '__main__':
    seed = 123
    epoch_num = 100
    data_num = 1000000
    threshold = 0.0
    dir_path = './figs/'
    fix_all_seed(seed=seed)
    run(epoch_num,threshold,data_num,dir_path)