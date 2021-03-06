import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import numpy as np

def nll_loss(scores, y):
    batch_size = y.size(1)
    scores = torch.reshape(scores, (y.numel(), -1))
    expscores = scores.exp()
    probabilities = expscores / expscores.sum(1, keepdim=True)
    answerprobs = probabilities[range(len(y.reshape(-1))), y.reshape(-1)]
    return torch.mean(-torch.log(answerprobs) * batch_size)

def evaluate_model(data, model, batch_size, device):
    with torch.no_grad():
        losses = []
        states = model.state_init(batch_size, device)
        for x, y in data:
            scores, states = model(x, states)  # scores <-> outputs
            loss = nll_loss(scores, y)
            # scores = Variable(scores).to(device)
            losses.append(loss.data.item()/batch_size)
    return np.exp(np.mean(losses))

def train(model, trn_dataset, val_dataset, tst_dataset, batch_size, sequence_length, lr, lr_factor, lr_change_epoch,
          max_grad_norm, device, variation, optimizer, epoch_num, checkpoints_dir_path, writer,
          latest_checkpoint_path=""):
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
    print(f"Epochs range = {epochs_range}")

    # epoch loop
    for e in epochs_range:
        states = model.state_init(batch_size, device)
        model.train()
        if e >= lr_change_epoch:
            print(f"Updating lr from {lr} to {lr / lr_factor}")
            lr /= lr_factor
            # for g in optimizer.param_groups:
            #     g['lr'] = lr
        running_loss = 0
        # batch loop
        for i, (x, y) in enumerate(trn_dataset):
            # move input and output to GPU
            if device.type == 'cuda':
                x = x.cuda()
                y = y.cuda()
                for state in states:
                    for param in state:
                        param = param.cuda()
            model.zero_grad()
            states = model.detach(states)
            scores, states = model(x, states)  ## tensor in the size of vocab_size over batch size (20 x 1 x 10,000)
            loss = nll_loss(scores, y)
            # compute gradients
            loss.backward()

            with torch.no_grad():
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                for param in model.parameters():
                    if not param.grad is None:
                        param -= lr * param.grad

            # accumulate loss
            running_loss += loss.item()

            if i % 500 == 0 and i > 0:
                print("Grad norm = {:.3f}, ".format(norm))
                print(f"Train : Batch num: {i}/{len(trn_dataset)}, Loss: {running_loss / i}")

        # deactivate dropout and batch normalization
        model.eval()
        perplexity_trn = evaluate_model(trn_dataset, model, batch_size, device)
        perplexity_val = evaluate_model(val_dataset, model, batch_size, device)
        perplexity_tst = evaluate_model(tst_dataset, model, batch_size, device)
        # save checkpoint at the end of each epoch
        curr_checkpoint_path = checkpoints_dir_path + f'/Zaremba-{variation}-{e}.pth'
        print(curr_checkpoint_path)
        torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / len(trn_dataset), },
                   curr_checkpoint_path)

        writer.add_scalars(f'{variation}', {
            'Train_Perplexity': float(perplexity_trn),
            'Val_Perplexity': float(perplexity_val),
            'Test_Perplexity': float(perplexity_tst)
        }, e)

        # print states at the end of each epoch
        print(f"Epoch {e}:")
        print(f"Training loss:     {running_loss / len(trn_dataset)}")
        print(f"Training Perplexity: {perplexity_trn}")
        print(f"Val Perplexity:      {perplexity_val}")
        print(f"Test Perplexity:     {perplexity_tst}")
        print(f"-----------------------------------------------")

