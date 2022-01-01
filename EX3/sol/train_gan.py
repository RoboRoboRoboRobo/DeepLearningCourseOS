import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

def train_gan(generator, discriminator, data_loader_train, batch_size, lr0, lam,
              device, generator_optimizer, discriminator_optimizer, epoch_num, checkpoints_dir_path, writer,
              num_of_disc_iter, variation, latest_checkpoint_path=""):
    # checkpoints handling
    if latest_checkpoint_path == "":
        first_epoch = 0
    else:
        # load previous training checkpoint
        checkpoint = torch.load(latest_checkpoint_path)
        generator.load_state_dict(checkpoint['model_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        generator.load_state_dict(checkpoint['model_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']

    # epoch iteration range
    epochs_range = range(first_epoch, epoch_num)
    print(f"Epochs range = {epochs_range}")
    # epoch loop

    lr = lr0
    for e in epochs_range:

        generator.train()
        discriminator.train()

        running_loss_dis = 0
        running_loss_gen = 0
        # batch loop
        for i, (x_real, _) in enumerate(data_loader_train):
            # move input and output to GPU
            if device.type == 'cuda':
                x = x.cuda()

            loss_dis, grad_penalty, grad_norm = train_discriminator(x_real, generator, discriminator, discriminator_optimizer,
                                            batch_size, lam, device, variation)

            # backward propagate and compute gradients
            if i % num_of_disc_iter == 0:
                loss_gen = train_generator(generator, discriminator, generator_optimizer,
                                           batch_size, device, variation)

            # accumulate loss
            running_loss_dis += loss_dis.item()
            running_loss_gen += loss_gen.item()

            if variation == "wgan":
                lr = (-lr0 / 1.0e05) * i + lr0
            with torch.no_grad():
                for param in discriminator.parameters():
                    if not param.grad is None:
                        param -= lr * param.grad
                for param in generator.parameters():
                    if not param.grad is None:
                        param -= lr * param.grad

            if i % 20 == 0 and i > 0:
                print("===========================")
                print(f"Train"
                      f"Batch num: {i}/{len(data_loader_train)}\n"
                      f"Loss Gen: {running_loss_gen / i}\n"
                      f"Loss Dis: {running_loss_dis / i}\n"
                      f"Grad penalty: {grad_penalty}\n"
                      f"Grad norm: {grad_norm[-1]}")

        # save checkpoint at the end of each epoch
        curr_checkpoint_path_gen = checkpoints_dir_path + f'/Gulrajani-{e}.pth'
        curr_checkpoint_path_dis = checkpoints_dir_path + f'/Gulrajani-{e}.pth'
        print(curr_checkpoint_path_gen)
        print(curr_checkpoint_path_dis)
        torch.save({'epoch': e, 'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': generator_optimizer.state_dict(),
                    'loss': running_loss_gen / len(data_loader_train), },
                    curr_checkpoint_path_gen)
        torch.save({'epoch': e, 'model_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': discriminator_optimizer.state_dict(),
                    'loss': running_loss_dis / len(data_loader_train), },
                    curr_checkpoint_path_dis)

        # writer.add_scalars(f'Kingsma', {
        #     'Train_Accuracy': float(acc_trn),
        #     'Test_Accuracy': float(acc_tst)
        # }, e)

        # print states at the end of each epoch
        print(f"Epoch {e}:")
        print(f"Training loss dis:     {running_loss_dis / len(data_loader_train)}")
        print(f"Training loss gen:     {running_loss_gen / len(data_loader_train)}")
        print(f"-----------------------------------------------")


def train_discriminator(x_real, generator, discriminator, discriminator_optimizer, batch_size, lam, device, variation):

    z = Variable(torch.randn((batch_size, generator.dim_z)))
    if device.type == 'cuda':
        z = z.cuda()
    x_gen = generator(z)

    prob_gen = discriminator(x_gen)
    x_real = x_real.view(-1, 1, int(x_real.shape[1]**0.5), int(x_real.shape[1]**0.5))
    prob_real = discriminator(x_real)

    discriminator_optimizer.zero_grad()
    if variation == "wgan":
        grad_penalty, grad_norm = gradient_penalty(discriminator, x_real, x_gen, device)
        loss = prob_gen.mean() - prob_real.mean() + lam * grad_penalty
    elif variation == "dcgan":
        loss = torch.mean(torch.log(prob_real) + torch.log(1-prob_gen))
        grad_penalty = "not_rel"
        grad_norm = "not_rel"

    loss.backward()

    discriminator_optimizer.step()

    return loss, grad_penalty, grad_norm


def train_generator(generator, discriminator, generator_optimizer, batch_size, device, variation):

    z = Variable(torch.randn((batch_size, generator.dim_z)))
    if device.type == 'cuda':
        z = z.cuda()
    x_gen = generator(z)

    prob_gen = discriminator(x_gen)

    generator_optimizer.zero_grad()
    if variation == "wgan":
        loss = - prob_gen.mean()
    elif variation == "dcgan":
        loss = torch.mean(torch.log(1-prob_gen))

    loss.backward()

    generator_optimizer.step()

    return loss


def gradient_penalty(discriminator, x_real, x_gen, device):

    batch_size = x_real.shape[0]
    eps = torch.rand(batch_size, 1, 1, 1)
    eps = eps.expand_as(x_real)
    if device.type == 'cuda':
        eps = eps.cuda()
    x_mixed = eps * x_real + (1-eps) * x_gen  # TODO check expand_as()
    x_mixed = Variable(x_mixed, requires_grad=True)

    prob_mixed = discriminator(x_mixed)

    gradients = torch_grad(outputs=prob_mixed, inputs=x_mixed,
                           grad_outputs=torch.ones(prob_mixed.size()).cuda() if device.type == "cuda" else torch.ones(
                               prob_mixed.size()),
                           create_graph=True, retain_graph=True)[0]


    # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    gradients = gradients.view(batch_size, -1)
    grad_norm = torch.norm(gradients, dim=-1) + 1e-12
    return torch.mean((grad_norm - 1)**2), grad_norm



