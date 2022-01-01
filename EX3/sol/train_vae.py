import torch


def elbo_loss(x, x_hat, kl_divergence):
    return torch.mean(torch.norm(x - x_hat, dim=-1)**2 + 0.1 * kl_divergence)

def evaluate_model(data, model, device):
    with torch.no_grad():
        mses = []
        for x, _ in data:
            x_hat = model(x)
            mses.append(torch.mean(torch.norm(x - x_hat, dim=-1)**2))
    return torch.mean(torch.Tensor(mses))**0.5

def train_vae(model, trn_dataset, tst_dataset, batch_size, lr,
          device, optimizer, epoch_num, checkpoints_dir_path, writer,
          lr_factor, lr_change_epoch, max_grad_norm,
          latest_checkpoint_path=""):
    # checkpoints handling
    if latest_checkpoint_path == "":
        first_epoch = 0
    else:
        print("\n******Loading latest checkpoint path******\n")
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
        if e >= lr_change_epoch:
            print(f"Updating lr from {lr} to {lr / lr_factor}")
            lr /= lr_factor
        model.train()

        running_loss = 0
        # batch loop
        for i, (x, y) in enumerate(trn_dataset):
            # move input and output to GPU
            if device.type == 'cuda':
                x = x.cuda()

            # backward propagate
            x_hat = model(x)
            L = elbo_loss(x, x_hat, model.kl_divergence)  # TODO consider  next(beta)
            # compute gradients
            L.backward()

            # accumulate loss
            running_loss += L.item()

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                for param in model.parameters():
                    if not param.grad is None:
                        param -= lr * param.grad

            if i % 500 == 0 and i > 0:
                print(f"Train : Batch num: {i}/{len(trn_dataset)}, Loss: {running_loss / i}")

        model.eval()
        acc_trn = evaluate_model(trn_dataset, model, device)
        acc_tst = evaluate_model(tst_dataset, model, device)
        # save checkpoint at the end of each epoch
        curr_checkpoint_path = checkpoints_dir_path + f'/Kingsma-{e}.pth'
        print(curr_checkpoint_path)
        torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / len(trn_dataset), },
                    curr_checkpoint_path)

        writer.add_scalars(f'Kingsma', {
            'Train_Accuracy': float(acc_trn),
            'Test_Accuracy': float(acc_tst)
        }, e)

        # print states at the end of each epoch
        print(f"Epoch {e}:")
        print(f"Training loss:     {running_loss / len(trn_dataset)}")
        print(f"Training Accuracy: {acc_trn}")
        print(f"Test Accuracy:     {acc_tst}")
        print(f"-----------------------------------------------")

