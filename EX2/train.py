import torch
from torch import nn

def train(model, variation, epoch_num, optimizer, data_loader_train, data_loader_test, checkpoints_dir_path,
    latest_checkpoint_path=""):
    train_acc_list = []
    test_acc_list = []
    epoch_list = []

    # checkpoints handling
    if latest_checkpoint_path is "":
        first_epoch = 0
    else:
        # load previous training checkpoint
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']

    # epoch iteration range
    epochs_range = range(first_epoch, epoch_num)
    print(f"epochs_range = {epochs_range}")

    # epoch loop
    for e in epochs_range:

        running_loss = 0

        # batch loop
        for images_batch, labels_batch in data_loader_train:
            # move input and output to GPU
            images_batch = Variable(images_batch).to(device)
            labels_batch = Variable(labels_batch).to(device)

            # clear gradients prior to new batch (since gradients are accumulated)
            optimizer.zero_grad()

            # obtain output probabilities (by feed forward images batch across the model)
            log_prob_output = model(images_batch)
            prob_output = torch.exp(log_prob_output)

            # obtain loss given log prbabilities and GT labels batch
            loss = criterion(prob_output, labels_batch)

            # compute gradients
            loss.backward()

            # update weights
            optimizer.step()

            # accumulate loss
            running_loss += loss.item()

        # deactivate dropout and batch normalization
        if ('DO' == variation):
            model.eval()

            # compute accuracies
        acc_train = evaluateModel(data_loader_train, model)
        acc_test = evaluateModel(data_loader_test, model)

        # accumulate accuracies
        train_acc_list.append(acc_train)
        test_acc_list.append(acc_test)
        epoch_list.append(e)

        # save checkpint at the end of each epoch
        curr_checkpoint_path = checkpoints_dir_path + f'/Lenet5-{variation}-{e}.pth'
        print(curr_checkpoint_path)
        torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / len(data_loader_train), },
                   curr_checkpoint_path)
        writer.add_scalars(f'{variation}', {
            'Train Acc': acc_train,
            'Test Acc': acc_test
        }, e)

        # save train and test accuracies figure after last epoch
        if e == epoch_num - 1:

            plt.figure()
            plt.plot(epoch_list, train_acc_list, label='Train Acc')
            plt.plot(epoch_list, test_acc_list, label='Test Acc')
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            var_type = "Original"
            if "DO" in variation:
                var_type = "Dropout"
            elif "WD" in variation:
                var_type = "Weight decay"
            elif "BN" in variation:
                var_type = "Batch Normalization"
            plt.title(f"Lenet5 - {var_type}")
            plt.legend()
            date_time = datetime.today().strftime('%d_%m_%Y_%H_%M_%S')
            str_figure = figures_dir_path + f'{variation}-{e}-{date_time}.png'
            plt.savefig(str_figure)

        # print states at the end of each epoch
        print(f"Epoch {e}:")
        print(f"Training loss:     {running_loss / len(data_loader_train)}")
        print(f"Training accuracy: {acc_train}")
        print(f"Test accuracy:     {acc_test}")
        print(f"-----------------------------------------------")
