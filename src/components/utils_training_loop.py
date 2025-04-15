def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (callable): Loss function.
        device (torch.device): Device to perform training on.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.

    Returns:
        tuple:
            - avg_loss (float): Average training loss for the epoch.
            - auc (float): Mean AUC score across all classes.
    """

    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for step, batch in pbar:

        if isinstance(batch['melspec'], list):
            batch_outputs = []
            batch_losses = []

            for i in range(len(batch['melspec'])):
                inputs = batch['melspec'][i].unsqueeze(0).to(device)
                target = batch['target'][i].unsqueeze(0).to(device)

                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()

                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())

            optimizer.step()
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch['target'].numpy()

        else:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())

        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, auc

def validate(model, loader, criterion, device):
    """
        Evaluates the model on the validation set.

        Args:
            model (torch.nn.Module): The model to evaluate.
            loader (torch.utils.data.DataLoader): DataLoader for validation data.
            criterion (callable): Loss function.
            device (torch.device): Device to perform validation on.

        Returns:
            tuple:
                - avg_loss (float): Average validation loss.
                - auc (float): Mean AUC score across all classes.
        """

    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if isinstance(batch['melspec'], list):
                batch_outputs = []
                batch_losses = []

                for i in range(len(batch['melspec'])):
                    inputs = batch['melspec'][i].unsqueeze(0).to(device)
                    target = batch['target'][i].unsqueeze(0).to(device)

                    output = model(inputs)
                    loss = criterion(output, target)

                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())

                outputs = torch.cat(batch_outputs, dim=0).numpy()
                loss = np.mean(batch_losses)
                targets = batch['target'].numpy()

            else:
                inputs = batch['melspec'].to(device)
                targets = batch['target'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, auc

def calculate_auc(targets, outputs):
    """
        Computes the mean AUC score across all classes.

        Args:
            targets (np.ndarray): Ground truth binary labels (shape: [n_samples, n_classes]).
            outputs (np.ndarray): Model outputs/logits (shape: [n_samples, n_classes]).

        Returns:
            float: Mean AUC score for all classes with at least one positive label.
        """

    num_classes = targets.shape[1]
    aucs = []

    probs = 1 / (1 + np.exp(-outputs))

    for i in range(num_classes):

        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)

    return np.mean(aucs) if aucs else 0.0