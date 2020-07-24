import time
import os
import torch
import torch.nn.functional as F


def run(model, train_loader, test_loader, epochs, optimizers, schedulers, writer, device):
    if len(optimizers) == 3:
        train_optimizers = optimizers[:-1]
        test_optimizer = optimizers[-1]
    else:
        train_optimizers = optimizers
        test_optimizer = None
    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, train_optimizers, train_loader, device)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, device, test_optimizer)
        for scheduler in schedulers:
            scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        if epoch % 10 == 0 or epoch == epochs:
            writer.save_checkpoint(model, optimizers, schedulers, epoch)


def train(model, optimizers, loader, device):
    model.train()

    if model.type == 'VAE':
        total_loss = {'train_kld': 0, 'train_rec': 0, 'train': 0}
    elif model.type == 'AE' or model.type == 'AD':
        if model.lam != -1:
            total_loss = {'train_rec': 0, 'train_reg': 0}
        else:
            total_loss = {'train_rec': 0}


    for (data, idx) in loader:
        for optimizer in optimizers:
            optimizer.zero_grad()
        x = data.x.to(device)
        if model.type == 'VAE':
            out, mu, logvar = model(x)
            reconstruciton_loss = F.l1_loss(out, x, reduction='mean')
            KL_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruciton_loss + model.lam * KL_div

            total_loss['train_kld'] += KL_div.item()
            total_loss['train_rec'] += reconstruciton_loss.item()
            total_loss['train'] += loss.item()
        elif model.type == 'AE':
            pred, z = model(x)
            if model.lam != -1:
                rec_loss = F.l1_loss(pred, x, reduction='mean')
                reg_loss = torch.mean(torch.norm(z))
                loss = rec_loss + model.lam * reg_loss
                total_loss['train_rec'] += rec_loss.item()
                total_loss['train_reg'] += reg_loss.item()
            else:
                loss = F.l1_loss(pred, x, reduction='mean')
                total_loss['train_rec'] += loss.item()
        elif model.type == 'AD':
            z = model.z_train[idx, :]
            pred = model(z)
            rec_loss = F.l1_loss(pred, x, reduction='mean')
            reg_loss = torch.mean(torch.norm(z))
            loss = rec_loss + model.lam * reg_loss
            total_loss['train_rec'] += rec_loss.item()
            total_loss['train_reg'] += reg_loss.item()

        else:
            raise ValueError('unexpected modeltype: {}'.format(model.type))

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    return {k: v/len(loader) for (k, v) in total_loss.items()}


def test(model, loader, device, test_optimizer):
    model.eval()

    if model.type == 'VAE':
        total_loss = {'test_kld': 0, 'test_rec': 0, 'test_map': 0, 'test': 0}
    elif model.type == 'AE' or model.type == 'AD':
        if model.lam != -1:
            total_loss = {'test_rec': 0, 'test_reg': 0}
        else:
            total_loss = {'test_rec': 0}

    with torch.no_grad():
        for i, (data, idx) in enumerate(loader):
            x = data.x.to(device)
            if model.type == 'VAE':
                pred, pred_map, mu, logvar = model(x, also_give_map=True)
                reconstruction_loss = F.l1_loss(pred, x, reduction='mean')
                reconstruction_loss_map = F.l1_loss(pred_map, x, reduction='mean')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                combined_loss = reconstruction_loss + model.lam * kld_loss
                total_loss['test_kld'] += kld_loss.item()
                total_loss['test_rec'] += reconstruction_loss.item()
                total_loss['test_map'] += reconstruction_loss_map.item()
                total_loss['test'] += combined_loss.item()
            elif model.type == 'AE':
                pred, z = model(x)
                if model.lam != -1:
                    total_loss['test_rec'] += F.l1_loss(pred, x, reduction='mean')
                    total_loss['test_reg'] += torch.mean(torch.norm(z))
                else:
                    total_loss['test_rec'] += F.l1_loss(pred, x, reduction='mean')
            elif model.type == 'AD':
                test_optimizer.zero_grad()
                z = model.z_test[idx, :]
                pred = model(z)
                total_loss['test_rec'] += F.l1_loss(pred, x, reduction='mean')
                total_loss['test_reg'] += torch.mean(torch.norm(z))
                loss = total_loss['test_rec'] + model.lam * total_loss['test_reg']
                loss.backward()
                test_optimizer.step()
            else:
                raise ValueError('unexpected model type {}'.format(model.type))


    return {k: v/len(loader) for (k, v) in total_loss.items()}


def eval_error(model, test_loader, device, mean, std, out_dir):
    model.eval()

    errors = []
    with torch.no_grad():
        for i, (data, idx) in enumerate(test_loader):
            x = data.x.to(device)
            if model.type == 'VAE':
                _, pred, _, _ = model(x, also_give_map=True)
            elif model.type == 'AE':
                pred, _ = model(x)
            elif model.type == 'AD':
                z = model.z_test[idx, :]
                pred = model(z)
            else:
                raise ValueError('Unexpected model type {}'.format(model.type))
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 1000
            reshaped_x *= 1000

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x)**2,
                          dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

        mean_error = new_errors.view((-1, )).mean()
        std_error = new_errors.view((-1, )).std()
        median_error = new_errors.view((-1, )).median()

    message = 'Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error,
                                                     median_error)

    out_error_fp = out_dir + '/euc_errors.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))
    print(message)
