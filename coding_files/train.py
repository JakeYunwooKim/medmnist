# This file is for training and saving models.
# If you need to confirm or modify some values, see "main.py".

import torch
import os
import pandas as pd
import time
import datetime
import torch.optim as optim
from main import time_of_tests, num_of_models, num_epochs, batch_size, n_channels, n_classes, data_flag, lr, weight_decay, train_loader, train_loader_at_eval, task, criterion
from medmnist import Evaluator
from mnist3dnet import Net1, Net2, Net3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

file_name_list = []
dir_name = f"./models/{time_of_tests:03}/"
eval_file_name_last = dir_name + f"eval_last.txt"
eval_file_name_best = dir_name + f"eval_best.txt"
params_file_name = dir_name + f"params_list.txt"
models_csv = dir_name + f"models_{time_of_tests:03}.csv"
create_dir(dir_name)

param_str = f"time of tests : {time_of_tests}\nnum of models : {num_of_models}\nepochs : {num_epochs}\nbatch size : {batch_size}"
with open(params_file_name, "w") as file:
    file.write(param_str)

# For each training, two kinds of models will be created and saved in here.
# last model : the final result at the end of training
# best model : the model with the best AUC score on the validation data
def train_model():
    for cnt in range(num_of_models):
        if cnt <= (num_of_models / 3):
            model = Net1(in_channels=n_channels, num_classes=n_classes).to(device)
        elif cnt <= (num_of_models * 2 / 3):
            model = Net2(in_channels=n_channels, num_classes=n_classes).to(device)
        else:
            model = Net3(in_channels=n_channels, num_classes=n_classes).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        model_df = pd.DataFrame([{'count' : f"{cnt:03}",
                                  'model' : model.__class__.__name__,
                                  'optimizer' : optimizer.__class__.__name__,
                                  'learning rate' : f"{lr}", 
                                  'weight decay' : f"{weight_decay}"}])
        
        valid_auc_max = 0
        best_epoch = 0
        model_file_name_best = dir_name + f"model_{time_of_tests:03}_{cnt:02}_best.pth"
        model_file_name_last = dir_name + f"model_{time_of_tests:03}_{cnt:02}_last.pth"

        # beginning the cnt-th training
        start_time = time.time()
        for epoch in range(1, num_epochs+1):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                outputs = model(inputs).to(device)
                
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32).to(device)
                    loss = criterion(outputs, targets).to(device)
                else:
                    targets = targets.squeeze().long().to(device)
                    loss = criterion(outputs, targets).to(device)
                
                loss.backward()
                optimizer.step()
            print("[cnt : {}, model : {}, epoch : {}]  Loss: {:.7f}".format(cnt+1, model.__class__.__name__, epoch, loss.item(), ))

            model.eval()

            y_true = torch.tensor([]).to(device)
            y_score = torch.tensor([]).to(device)

            # checking the epoch-th model on the validation data
            # updating the best model if necessary
            with torch.no_grad():
                for inputs, targets in train_loader_at_eval:
                    inputs = inputs.to(device)
                    outputs = model(inputs).to(device)

                    if task == 'multi-label, binary-class':
                        targets = targets.to(torch.float32).to(device)
                        outputs = outputs.softmax(dim=-1).to(device)
                    else:
                        targets = targets.squeeze().long().to(device)
                        outputs = outputs.softmax(dim=-1).to(device)
                        targets = targets.float().resize_(len(targets), 1).to(device)

                    y_true = torch.cat((y_true, targets), 0).to(device)
                    y_score = torch.cat((y_score, outputs), 0).to(device)

                y_true = y_true.cpu().numpy()
                y_score = y_score.detach().cpu().numpy()

                evaluator = Evaluator(data_flag, 'val')
                valid_auc, valid_acc = evaluator.evaluate(y_score)

                print('valid auc : %f  acc : %f' % (valid_auc, valid_acc))

                if valid_auc >= valid_auc_max:
                    print(f'\t### ACC score for valid data has increased: ({valid_auc_max:.7f} --> {valid_auc:.7f}). The best model is updated.')
                    torch.save(model.state_dict(), model_file_name_best)
                    valid_auc_max = valid_auc
                    best_epoch = epoch
        end_time = time.time()
        total_time_str = str(datetime.timedelta(seconds=end_time-start_time))

        # saving the last model
        torch.save(model.state_dict(), model_file_name_last)

        if not os.path.exists(models_csv):
            model_df.to_csv(models_csv, index=False, mode='w')
        else:
            model_df.to_csv(models_csv, index=False, header=False, mode='a')

        # evaluating the last model
        with torch.no_grad():
            y_true = torch.tensor([]).to(device)
            y_score = torch.tensor([]).to(device)

            for inputs, targets in train_loader_at_eval:
                inputs = inputs.to(device)
                outputs = model(inputs).to(device)

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32).to(device)
                    outputs = outputs.softmax(dim=-1).to(device)
                else:
                    targets = targets.squeeze().long().to(device)
                    outputs = outputs.softmax(dim=-1).to(device)
                    targets = targets.float().resize_(len(targets), 1).to(device)

                y_true = torch.cat((y_true, targets), 0).to(device)
                y_score = torch.cat((y_score, outputs), 0).to(device)

            y_true = y_true.cpu().numpy()
            y_score = y_score.detach().cpu().numpy()
            evaluator = Evaluator(data_flag, 'val')
            auc, acc = evaluator.evaluate(y_score)
        
            print('==> Evaluating the last model ...')
            
            print(f"valid auc : {auc:.07f}   acc : {acc:.07f}")

            with open(eval_file_name_last, 'a') as file:
                file.write(f"cnt : {cnt:03}   last auc : {auc:.07f}   acc : {acc:.07f}   training time : " + total_time_str)
                if cnt <= num_of_models:
                    file.write('\n')

        # evaluating the best model        
        model.load_state_dict(torch.load(model_file_name_best))

        with torch.no_grad():
            y_true = torch.tensor([]).to(device)
            y_score = torch.tensor([]).to(device)

            for inputs, targets in train_loader_at_eval:
                inputs = inputs.to(device)
                outputs = model(inputs).to(device)

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32).to(device)
                    outputs = outputs.softmax(dim=-1).to(device)
                else:
                    targets = targets.squeeze().long().to(device)
                    outputs = outputs.softmax(dim=-1).to(device)
                    targets = targets.float().resize_(len(targets), 1).to(device)

                y_true = torch.cat((y_true, targets), 0).to(device)
                y_score = torch.cat((y_score, outputs), 0).to(device)

            y_true = y_true.cpu().numpy()
            y_score = y_score.detach().cpu().numpy()
            evaluator = Evaluator(data_flag, 'val')
            auc, acc = evaluator.evaluate(y_score)
        
            print('the best epoch :', best_epoch)
            print('==> Evaluating the best model ...')
            
            print(f"valid auc : {auc:.07f}   acc : {acc:.07f}")

            with open(eval_file_name_best, 'a') as file:
                file.write(f"cnt : {cnt:03}   best auc : {auc:.07f}   acc : {acc:.07f}   epochs : {best_epoch:03}")
                if cnt <= num_of_models:
                    file.write('\n')

if __name__ == "main":
    train_model()
    
    print("Training finished.")
