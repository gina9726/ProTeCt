### engine.py
# Functions for train/test an epoch.
###

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from metrics import averageMeter, multilabel_accuracy, accuracy, hierConsistency
import pdb


# train one single epoch
def train_one_epoch(model, optimizer, sched, data_loader, param_names, device, epoch, cfg, args, treecut_generator=None, \
        leaf_nodes=None, intnl_nodes=None, sublabels=None):
    # setup average meters
    meter = {}
    meter["loss"] = averageMeter()
    meter["acc"] = averageMeter()
    if intnl_nodes is not None:
        meter["loss_node"] = averageMeter()
        meter.update({f'n{i}_acc': averageMeter() for i in range(len(intnl_nodes))})
        if cfg['data']['sampler']['name'] == 'nshot':
            meter['consistency'] = hierConsistency(data_loader.sampler.sampled_idx, len(data_loader.dataset.indices))
        else:
            meter['consistency'] = hierConsistency(data_loader.dataset.indices)

    print_freq = cfg["print_freq"]
    if cfg['model']['name'] in ['RN50', 'RN101']:
        model.eval()
    else:
        model.train()
    random_cut = True if treecut_generator is not None else False
    if random_cut:
        treecut_generator.train()
        df = pd.DataFrame(param_names)

    # train 1 epoch
    for (step, value) in tqdm(enumerate(data_loader)):
        image = value[0].cuda()
        target = value[1].cuda()
        index = value[2]
        bz = target.shape[0]

        # classification at each internal node
        loss_node = 0
        if intnl_nodes is not None:
            if cfg['loss'].get('lambda', 0) == 0:
                with torch.no_grad():
                    loss_node = []
                    for i, x in enumerate(intnl_nodes):
                        tgt_mapping = sublabels[:, i]
                        ntarget = tgt_mapping[target].long()
                        idx = (ntarget >= 0)
                        if sum(idx) > 0:
                            # forward
                            ntarget, nindex = ntarget[idx], index[idx]
                            param_idx = torch.tensor(x).cuda()
                            text = list(df.iloc[param_idx.cpu()][0])
                            nlogits = model(image[idx], text)['logits_per_image']

                            # compute loss and accuracy
                            nloss = F.cross_entropy(nlogits, ntarget)
                            loss_node.append(nloss)
                            nconf, npred = torch.softmax(nlogits, dim=-1).max(dim=-1)
                            niscorrect = (npred == ntarget)
                            nacc = niscorrect.float().mean() * 100.0
                            meter[f'n{i}_acc'].update(nacc.item(), sum(idx))
                            meter['consistency'].update(nindex, niscorrect)
                    loss_node = torch.stack(loss_node).mean()
                    meter['loss_node'].update(loss_node.item(), bz)
            else:
                loss_node = []
                for i, x in enumerate(intnl_nodes):
                    tgt_mapping = sublabels[:, i]
                    ntarget = tgt_mapping[target].long()
                    idx = (ntarget >= 0)
                    if sum(idx) > 0:
                        # forward
                        ntarget, nindex = ntarget[idx], index[idx]
                        param_idx = torch.tensor(x).cuda()
                        text = list(df.iloc[param_idx.cpu()][0])
                        nlogits = model(image[idx], text)['logits_per_image']

                        # compute loss and accuracy
                        nloss = F.cross_entropy(nlogits, ntarget)
                        loss_node.append(nloss)
                        nconf, npred = torch.softmax(nlogits, dim=-1).max(dim=-1)
                        niscorrect = (npred == ntarget)
                        nacc = niscorrect.float().mean() * 100.0
                        meter[f'n{i}_acc'].update(nacc.item(), sum(idx))
                        meter['consistency'].update(nindex, niscorrect)
                loss_node = torch.stack(loss_node).mean()
                meter['loss_node'].update(loss_node.item(), bz)

        # classification at leaf nodes
        if random_cut:
            label_set, tgt_mapping = treecut_generator.get_randomcut()
            param_idx = torch.where(label_set)[0]
            target = tgt_mapping[target]
            text = list(df.iloc[param_idx.cpu()][0])
            output = model(image, text)
        else:
            output = model(image)

        leaf_logits = output['logits_per_image']
        n_classes = leaf_logits.shape[1]
        if intnl_nodes is not None:
            conf, pred = torch.softmax(leaf_logits, dim=-1).max(dim=-1)
            iscorrect = (pred == target)
            meter['consistency'].update(index, iscorrect)

        if cfg["loss"]["name"] == "ce":
            loss = F.cross_entropy(leaf_logits, target)
            acc = accuracy(leaf_logits, target)[0].item() # assuming top 1 acc
        elif cfg["loss"]["name"] == "bce":
            # need to find all the root path of the node
            y = torch.zeros(bz, n_classes).to(device)
            y[range(bz), target] = 1
            assert y.shape == leaf_logits.shape # both the shape should be (bz, num label set)
            loss = F.binary_cross_entropy_with_logits(leaf_logits, y)

            # computer accuracy
            prob = torch.sigmoid(leaf_logits)
            prob = 1.0 * (prob > 0.5)
            # compute the metric
            acc = multilabel_accuracy(prob.cpu(), y.cpu())

        loss += cfg['loss'].get('lambda', 0) * loss_node

        # update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # lr update
        if "use_scheduler" in cfg["optim"] and cfg["optim"]["use_scheduler"] == True:
            sched.step()

        # update meters
        meter["loss"].update(loss.item(), bz)
        meter["acc"].update(acc, bz)

        # print information
        if (step+1) % print_freq == 0 or args.debug:
            print(f"[Train] Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Acc: {acc}")

        # debug mode
        if args.debug:
            break

    meter = {k:meter[k].avg for k in meter.keys()}
    return meter



# eval one single epoch
@torch.no_grad()
def eval_one_epoch(model, data_loader, param_names, device, epoch, cfg, args, leaf_nodes=None, intnl_nodes=None, sublabels=None):
    # setup average meters
    meter = {}
    meter["loss"] = averageMeter()
    meter["acc"] = averageMeter()
    if intnl_nodes is not None:
        meter["loss_node"] = averageMeter()
        meter.update({f'n{i}_acc': averageMeter() for i in range(len(intnl_nodes))})
        meter['consistency'] = hierConsistency(data_loader.dataset.indices)

    print_freq = cfg["print_freq"]
    model.eval()
    df = pd.DataFrame(param_names)

    for (step, value) in tqdm(enumerate(data_loader)):
        image = value[0].cuda()
        target = value[1].cuda()
        index = value[2]
        bz = target.shape[0]

        # classification at each internal node
        loss_node = 0
        if intnl_nodes is not None:
            loss_node = []
            for i, x in enumerate(intnl_nodes):
                tgt_mapping = sublabels[:, i]
                ntarget = tgt_mapping[target].long()
                idx = (ntarget >= 0)
                if sum(idx) > 0:
                    # forward
                    ntarget, nindex = ntarget[idx], index[idx]
                    param_idx = torch.tensor(x).cuda()
                    text = list(df.iloc[param_idx.cpu()][0])
                    nlogits = model(image[idx], text)['logits_per_image']

                    # compute loss and accuracy
                    nloss = F.cross_entropy(nlogits, ntarget)
                    loss_node.append(nloss)
                    nconf, npred = torch.softmax(nlogits, dim=-1).max(dim=-1)
                    niscorrect = (npred == ntarget)
                    nacc = niscorrect.float().mean() * 100.0
                    meter[f'n{i}_acc'].update(nacc.item(), sum(idx))
                    meter['consistency'].update(nindex, niscorrect)
            loss_node = torch.stack(loss_node).mean()
            meter['loss_node'].update(loss_node.item(), bz)

        # classification at leaf nodes
        text = list(df.iloc[leaf_nodes][0])
        output = model(image, text)

        leaf_logits = output['logits_per_image']
        n_classes = leaf_logits.shape[1]
        if intnl_nodes is not None:
            conf, pred = torch.softmax(leaf_logits, dim=-1).max(dim=-1)
            iscorrect = (pred == target)
            meter['consistency'].update(index, iscorrect)

        if cfg["loss"]["name"] == "ce":
            loss = F.cross_entropy(leaf_logits, target)
            acc = accuracy(leaf_logits, target)[0].item() # assuming top 1 acc
        elif cfg["loss"]["name"] == "bce":
            # need to find all the root path of the node
            y = torch.zeros(bz, n_classes).to(device)
            y[range(bz), target] = 1
            assert y.shape == leaf_logits.shape # both the shape should be (bz, num label set)
            loss = F.binary_cross_entropy_with_logits(leaf_logits, y)

            # computer accuracy
            prob = torch.sigmoid(leaf_logits)
            prob = 1.0 * (prob > 0.5)
            # compute the metric
            acc = multilabel_accuracy(prob.cpu(), y.cpu())

        loss += cfg['loss'].get('lambda', 0) * loss_node

        # update meters
        meter["loss"].update(loss.item(), bz)
        meter["acc"].update(acc, bz)

        # print information
        if (step+1) % print_freq == 0 or args.debug:
            print(f"[Eval] Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Acc: {acc}")

        # debug mode
        if args.debug:
            break

    meter = {k:meter[k].avg for k in meter.keys()}
    return meter


