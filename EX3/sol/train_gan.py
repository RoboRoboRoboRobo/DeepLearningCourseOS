import torch
from torch.autograd import Variable
import os
from torchvision import utils

def train_gan(generator, discriminator, data_loader_train, batch_size, lr0, lam,
              device, generator_optimizer, discriminator_optimizer, epoch_num, checkpoints_dir_path, writer,
              num_of_disc_iter, variation, latest_checkpoint_path=""):

    # epoch iteration range
    epochs_range = range(0, epoch_num)
    print(f"Epochs range = {epochs_range}")
    # epoch loop

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    if device == 'cuda':
        one = one.cuda()
        mone = mone.cuda()

    count_gen_iterations = 0

    for e in epochs_range:
        # batch loop
        for i, (x_real, _) in enumerate(data_loader_train):
            if i == data_loader_train.dataset.__len__() // batch_size:
                break
            if variation == 'wgan':
                for p in discriminator.parameters():
                    p.requires_grad = True

                # move input and output to GPU
                if device.type == 'cuda':
                    x_real = x_real.cuda()
                loss_dis_gen, loss_dis_real, grad_penalty, grad_norm = train_discriminator(x_real, mone, one, generator, discriminator, discriminator_optimizer,
                                                batch_size, lam, device, variation)
                if count_gen_iterations % 20 == 0 and i > 0:
                    print("===========================")
                    print(f"Train"
                          f"Generator iteration: {count_gen_iterations}\n"
                          f"Loss dis gen: {loss_dis_gen}\n"
                          f"Loss dis real: {loss_dis_real}\n"
                          f"Grad penalty: {grad_penalty}\n"
                          f"Grad norm: {grad_norm[-1]}")
            elif variation == 'dcgan':
                loss_dis = train_discriminator(x_real, mone, one, generator, discriminator, discriminator_optimizer,
                                                batch_size, lam, device, variation)

            # backward propagate and compute gradients
            if i % num_of_disc_iter == 0:
                count_gen_iterations += 1
                for p in discriminator.parameters():
                    p.requires_grad = False  # to avoid computation
                loss_gen, gen_cost = train_generator(generator, mone, one, discriminator, generator_optimizer,
                                           batch_size, device, variation)
                if count_gen_iterations % 20 == 0 and i > 0:
                    print("===========================")
                    print(f"Train Gen"
                          f"Generator iteration: {count_gen_iterations}\n"
                          f"Loss gen: {loss_gen}\n"
                          f"Gen cost: {gen_cost}\n")

            if count_gen_iterations % 100 == 0:
                torch.save(generator, f"./generator-{e}.pkl")
                torch.save(discriminator, f"./discriminator-{e}.pkl")
                print('Models save to ./generator.pkl & ./discriminator.pkl ')
                if not os.path.exists('training_result_images/'):
                    os.makedirs('training_result_images/')

                z = Variable(torch.randn(batch_size, generator.dim_z, 1, 1))
                if device.type == 'cuda':
                    z = z.cuda()
                x_gen = generator(z)
                x_gen = x_gen.mul(0.5).add(0.5)
                x_gen = x_gen.data.cpu()[:64]
                grid = utils.make_grid(x_gen)
                utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(count_gen_iterations).zfill(3)))

        # save checkpoint at the end of each epoch
        curr_checkpoint_path_gen = checkpoints_dir_path + f'/Gulrajani-{e}.pth'
        curr_checkpoint_path_dis = checkpoints_dir_path + f'/Gulrajani-{e}.pth'
        print(curr_checkpoint_path_gen)
        print(curr_checkpoint_path_dis)
        torch.save({'epoch': e, 'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': generator_optimizer.state_dict(),
                    'loss': loss_gen, },
                    curr_checkpoint_path_gen)
        torch.save({'epoch': e, 'model_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': discriminator_optimizer.state_dict(),
                    'loss': loss_dis_gen, },
                    curr_checkpoint_path_dis)

        # writer.add_scalars(f'Kingsma', {
        #     'Train_Accuracy': float(acc_trn),
        #     'Test_Accuracy': float(acc_tst)
        # }, e)

def train_discriminator(x_real, mone, one, generator, discriminator, discriminator_optimizer, batch_size, lam, device, variation):
    discriminator_optimizer.zero_grad()

    z = Variable(torch.randn(batch_size, generator.dim_z, 1, 1))
    if device.type == 'cuda':
        z = z.cuda()

    x_gen = generator(z)
    prob_real = discriminator(x_real)
    prob_gen = discriminator(x_gen)
    # x_real = x_real.view(-1, 1, int(x_real.shape[1]**0.5), int(x_real.shape[1]**0.5))

    if variation == "wgan":
        grad_penalty, grad_norm = gradient_penalty(discriminator, x_real, x_gen, device)

        loss_real = prob_real.mean()
        loss_real.backward(mone)

        loss_gen = prob_gen.mean()
        loss_gen.backward(one)

        loss_grad_pen = lam * grad_penalty
        loss_grad_pen.backward()
        discriminator_optimizer.step()
        return loss_gen, loss_real, grad_penalty, grad_norm[-1][-1]

    elif variation == "dcgan":
        real_labels = torch.ones(batch_size)
        fake_labels = torch.zeros(batch_size)

        if device.type == 'cuda':
            real_labels, fake_labels = Variable(real_labels).cuda(), Variable(fake_labels).cuda()
        else:
            real_labels, fake_labels = Variable(real_labels), Variable(fake_labels)

        d_loss_real = discriminator.loss(prob_real.flatten(), real_labels)
        d_loss_fake = discriminator.loss(prob_gen.flatten(), fake_labels)

        # Optimize discriminator
        d_loss = d_loss_real + d_loss_fake
        discriminator.zero_grad()
        d_loss.backward()
        discriminator_optimizer.step()

        return d_loss

def train_generator(generator, mone, ones, discriminator, generator_optimizer, batch_size, device, variation):
    generator_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, generator.dim_z, 1, 1))
    if device.type == 'cuda':
        z = z.cuda()
    x_gen = generator(z)
    prob_gen = discriminator(x_gen)

    if variation == "wgan":
        loss = prob_gen.mean()
        loss.backward(mone)
        g_cost = -loss
        generator_optimizer.step()
        return loss, g_cost
    elif variation == "dcgan":
        real_labels = torch.ones(batch_size)
        fake_labels = torch.zeros(batch_size)

        if device.type == 'cuda':
            real_labels, fake_labels = Variable(real_labels).cuda(), Variable(fake_labels).cuda()
        else:
            real_labels, fake_labels = Variable(real_labels), Variable(fake_labels)

        g_loss = generator.loss(prob_gen.flatten(), real_labels)
        discriminator.zero_grad()
        generator.zero_grad()
        g_loss.backward()
        generator_optimizer.step()
        return g_loss


def gradient_penalty(discriminator, x_real, x_gen, device):

    batch_size = x_real.shape[0]
    eps = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
    eps = eps.expand(batch_size, discriminator.dim, discriminator.image_size[0], discriminator.image_size[0])
    if device.type == 'cuda':
        eps = eps.cuda()
    x_mixed = eps * x_real + ((1 - eps) * x_gen)  # TODO check expand_as()
    if device.type == 'cuda':
        x_mixed = x_mixed.cuda()
    x_mixed = Variable(x_mixed, requires_grad=True)

    prob_mixed = discriminator(x_mixed)

    gradients = torch.autograd.grad(outputs=prob_mixed, inputs=x_mixed,
                           grad_outputs=torch.ones(prob_mixed.size()).cuda() if device.type == "cuda" else torch.ones(
                               prob_mixed.size()),
                           create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    grad_norm = torch.norm(gradients, dim=-1)
    return grad_penalty, grad_norm



