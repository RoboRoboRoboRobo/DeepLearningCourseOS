import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import numpy as np


def evaluateModel(dataset, model, criterion, device):
    with torch.no_grad():
        # losses = []
        loss = 0
        for i, (x, y) in enumerate(dataset):
            outputs = model(x)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            outputs = Variable(outputs).to(device)
            y = y.reshape(-1)
            loss += torch.exp(criterion(outputs, y))
            if i % 500 == 0:
                print(f"Finished batch number {i}/{len(dataset)} in evaluation")

        return loss / len(dataset)


def train(model, trn_dataset, val_dataset, tst_dataset, batch_size, sequence_length, lr, lr_factor, lr_change_epoch,
          max_grad_norm, embed, device, variation, optimizer, epoch_num, checkpoints_dir_path, writer,
          latest_checkpoint_path=""):
    train_acc_list = []
    test_acc_list = []
    epoch_list = []
    criterion = nn.CrossEntropyLoss()

    # checkpoints handling
    if latest_checkpoint_path == "":
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
        model.train()
        if e >= lr_change_epoch:
            lr /= lr_factor
            for g in optimizer.param_groups:
                g['lr'] = lr
        running_loss = 0

        # batch loop
        # for (x, y) in trn_dataset:
        for i, (x, y) in enumerate(trn_dataset):  ## TODO move back
            # move input and output to GPU
            x = Variable(x).to(device)
            y = Variable(y).to(device)

            # clear gradients prior to new batch (since gradients are accumulated)
            # optimizer.zero_grad()

            # obtain output probabilities (by feed forward images batch across the model)
            # x = embed(x) ## seq_len x batch_size x word_vec_size (35x20x200)
            # y = embed(y) ## seq_len x batch_size x word_vec_size (35x20x200)
            model.zero_grad()
            x = model(x)  ## tensor in the size of vocab_size over batch size (20 x 1 x 10,000)

            # obtain loss
            x = torch.exp(x)
            x = x.view(-1, model.embed.vocab_size)
            y = y.reshape(batch_size * sequence_length)
            loss = criterion(x, y)

            # compute gradients
            loss.backward()

            # update weights
            with torch.no_grad():
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                for param in model.parameters():
                    if param is None:
                        param -= lr * param.grad

            # accumulate loss
            running_loss += loss.item()

            if i % 500 == 0 and i > 0:
                print("dw.norm() = {:.3f}, ".format(norm))
                print(f"Train : batch num: {i}/{len(trn_dataset)} loss: {running_loss / i}")

        # deactivate dropout and batch normalization
        if ('DO' in variation):
            model.eval()
            # compute accuracies
        ## TODO consider eval
        perplexity_trn = evaluateModel(trn_dataset, model, criterion, device)
        perplexity_val = evaluateModel(val_dataset, model, criterion, device)
        perplexity_tst = evaluateModel(tst_dataset, model, criterion, device)
        #
        # save checkpoint at the end of each epoch
        curr_checkpoint_path = checkpoints_dir_path + f'/Zaremba-{variation}-{e}.pth'
        print(curr_checkpoint_path)
        torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / len(trn_dataset), },
                   curr_checkpoint_path)
        writer.add_scalars(f'{variation}', {
            'Train Perplexity': perplexity_trn,
            'Val Perplexity': perplexity_val,
            'Test Perplexity': perplexity_tst
        }, e)

        # print states at the end of each epoch
        print(f"Epoch {e}:")
        print(f"Training loss:     {running_loss / len(trn_dataset)}")
        print(f"Training Perplexity: {perplexity_trn}")
        print(f"Val Perplexity:      {perplexity_val}")
        print(f"Test Perplexity:     {perplexity_tst}")
        print(f"-----------------------------------------------")

