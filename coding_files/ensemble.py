# This file is for ensembling models with weights.
# IMPOSSIBLE to use this file BEFORE RUNNING "train.py".

from main import n_channels, n_classes, test_loader, num_of_models, time_of_tests, task, data_flag
from train import models_csv, dir_name
from mnist3dnet import Net1, Net2, Net3
from medmnist import Evaluator
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loading the saved model
models_last_list = []
models_best_list = []
models_name_list = pd.read_csv(models_csv)['model'].values.tolist()

for i in range(num_of_models):
    if models_name_list[i] == "Net1":
        model = Net1(in_channels=n_channels, num_classes=n_classes).to(device)
    elif models_name_list[i] == "Net2":
        model = Net2(in_channels=n_channels, num_classes=n_classes).to(device)
    else:
        model = Net3(in_channels=n_channels, num_classes=n_classes).to(device)
    
    model.load_state_dict(torch.load(dir_name + f"model_{time_of_tests:03}_{i:02}_last.pth"))
    models_last_list.append(model)
    del model

for i in range(num_of_models):
    if models_name_list[i] == "Net1":
        model = Net1(in_channels=n_channels, num_classes=n_classes).to(device)
    elif models_name_list[i] == "Net2":
        model = Net2(in_channels=n_channels, num_classes=n_classes).to(device)
    else:
        model = Net3(in_channels=n_channels, num_classes=n_classes).to(device)
    
    model.load_state_dict(torch.load(dir_name + f"model_{time_of_tests:03}_{i:02}_best.pth"))
    models_best_list.append(model)
    del model

# MODIFY the values of lists to decide weights of models in here
# consider "eval_last.txt" and "eval_best.txt" files in "./models/{time_of_tests}/"
weight_of_last_models = [1 for i in range(num_of_models)]
weight_of_last_models[1] = 0.5
weight_of_last_models[3] = 1.5

weight_of_best_models = [1 for i in range(num_of_models)]
weight_of_best_models[5] = 0.8
weight_of_best_models[0] = 1.1
weight_of_best_models[1] = 1.1

# adjust weights between two groups of models in here
weight_of_last_models = [w * 0.5 for w in weight_of_last_models]
weight_of_best_models = [w * 1.5 for w in weight_of_best_models]

# evaluating the ensembled results
print('==> Evaluating ...')

with torch.no_grad():
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    for i in range(len(models_last_list)):
        models_last_list[i].eval()
    for i in range(len(models_best_list)):
        models_best_list[i].eval()

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs_last_list = [models_last_list[i](inputs) * weight_of_last_models[i] for i in range(num_of_models)]
        outputs_best_list = [models_best_list[i](inputs) * weight_of_best_models[i] for i in range(num_of_models)]
        outputs = ((sum(outputs_last_list) + sum(outputs_best_list)) / (sum(weight_of_last_models) + sum(weight_of_best_models))).to(device)

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            outputs = outputs.softmax(dim=-1).to(device)
        else:
            targets = targets.squeeze().long().to(device)
            outputs = outputs.softmax(dim=-1).to(device)
            targets = targets.float().resize_(len(targets), 1).to(device)

        y_true = torch.cat((y_true, targets), 0).to(device)
        y_score = torch.cat((y_score, outputs), 0).to(device)
        _, y_pred = torch.max(y_score.data, 1)
    
    y_true = y_true.cpu().numpy()
    y_score = y_score.detach().cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    evaluator = Evaluator(data_flag, 'test')
    auc, acc = evaluator.evaluate(y_score)
    cm = confusion_matrix(y_true.tolist(), y_pred.tolist())

print(f'auc of the final test : {auc:.7f}')
print(f'acc of the final test : {acc:.7f}')
print(cm)

# recording the result in a text file
with open("summary_of_ensemble.txt", 'a') as file:
    file.write(f"time of tests : {time_of_tests:03}\n")
    file.write(f"weight of last models : {list(weight_of_last_models)}\n")
    file.write(f"weight of best models : {list(weight_of_best_models)}\n")
    file.write(f"auc : {auc:.07f}   acc : {acc:.07f}\n")
    file.write("confusion matrix :\n")
    file.write(f"{cm}")
    file.write("\n========\n")
