import time
import os
import torch
import torch.nn.functional as F

lam = 0.001
runVAE = True


def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer, device):
    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, optimizer, train_loader, device)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, device)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)


def train(model, optimizer, loader, device):
    model.train()

    if runVAE:
        total_loss = {'train_kld': 0, 'train_rec': 0, 'train_combined': 0}
    else:
        total_loss = {'train_rec': 0}

    for data in loader:
        optimizer.zero_grad()
        x = data.x.to(device)
        if runVAE:
            out, mu, logvar = model(x)
            reconstruciton_loss = F.l1_loss(out, x, reduction='mean')
            KL_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruciton_loss + lam * KL_div

            total_loss['train_kld'] += KL_div.item()
            total_loss['train_rec'] += reconstruciton_loss.item()
            total_loss['train_combined'] += loss.item()
        else:
            out = model(x)
            loss = F.l1_loss(out, x, reduction='mean')
            total_loss['train_rec'] += loss.item()
        loss.backward()
        optimizer.step()

    return {k: v/len(loader) for (k, v) in total_loss.items()}


def test(model, loader, device):
    model.eval()

    if runVAE:
        total_loss = {'test_kld': 0, 'test_rec': 0, 'test_map': 0, 'test': 0}
    else:
        total_loss = {'test_rec': 0}

    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            pred, pred_map, mu, logvar = model(x, also_give_map=True)
            if runVAE:
                reconstruction_loss = F.l1_loss(pred, x, reduction='mean')
                reconstruction_loss_map = F.l1_loss(pred_map, x, reduction='mean')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                combined_loss = reconstruction_loss + lam * kld_loss
                total_loss['test_kld'] += kld_loss.item()
                total_loss['test_rec'] += reconstruction_loss.item()
                total_loss['test_map'] += reconstruction_loss_map.item()
                total_loss['test'] += combined_loss.item()
            else:
                total_loss['test_rec'] += F.l1_loss(pred, x, reduction='mean')

    return {k: v/len(loader) for (k, v) in total_loss.items()}


def eval_error(model, test_loader, device, meshdata, out_dir):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            _, pred_map, _, _ = model(x, also_give_map=True)
            num_graphs = data.num_graphs
            reshaped_pred = (pred_map.view(num_graphs, -1, 3).cpu() * std) + mean
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
