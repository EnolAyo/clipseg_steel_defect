import torch
import inspect
import json
import yaml
import math
import os
import sys
from datetime import datetime
from datasets.steel_dataset import SeverstalDataset
from general_utils import log
from torchvision import transforms
import numpy as np
from functools import partial
from os.path import expanduser, join, isfile, basename

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext
from torch.utils.data import DataLoader

from general_utils import TrainingLogger, get_attribute, filter_args, log, training_config_from_cli_args

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + 1) / (pred.sum() + target.sum() + 1)

def validate(model, dataset, config):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    metric_class, use_metric = config.val_metric_class, config.use_val_metric
    loss_fn = get_attribute(config.loss)

    model.eval()
    model.cuda()

    if metric_class is not None:
        metric = get_attribute(metric_class)()

    with torch.no_grad():

        i, losses = 0, []
        for data_x, data_y in data_loader:

            data_x = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_x]
            data_y = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_y]

            # prompts = model.sample_prompts(data_x[1], prompt_list=('a photo of a {}',))
            prompts = data_x[1]
            pred, visual_q, _, _ = model(data_x[0], prompts, return_features=True)

            if metric_class is not None:
                metric.add([pred], data_y)

            # pred = model(data_x[0], prompts)
            # loss = loss_fn(pred[0], data_y[0])
            loss = loss_fn(pred, data_y[0]) + dice_loss(pred, data_y[0])
            losses += [float(loss)]

            i += 1

            if config.val_max_iterations is not None and i > config.val_max_iterations:
                break

    if use_metric is None:
        return np.mean(losses), {}, False
    else:
        metric_scores = {m: s for m, s in zip(metric.names(), metric.value())} if metric is not None else {}
        return np.mean(losses), metric_scores, True


def main():
    log_tensorboard = os.path.join("logs_tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))

    config = training_config_from_cli_args()

    val_interval, best_val_loss, best_val_score = config.val_interval, float('inf'), float('-inf')

    model_cls = get_attribute(config.model)
    _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)
    model = model_cls(**model_args).cuda()

    dataset_cls = get_attribute(config.dataset)
    _, dataset_args, _ = filter_args(config, inspect.signature(dataset_cls).parameters)

    dataset = dataset_cls(**dataset_args)
    json_path = './Severstal/annotations_COCO.json'
    images_path = './Severstal/train_subimages'
    mean = [0.34388125, 0.34388125, 0.34388125]
    std = [0.13965334, 0.13965334, 0.13965334]
    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = SeverstalDataset(json_path=json_path, images_path=images_path, image_size=(256,256),
                               n_support=1, transform=transform, split='train')
    val_dataset = SeverstalDataset(json_path=json_path, images_path=images_path, image_size=(256,256),
                               n_support=1, transform=transform, split='val')
    log.info(f'Train dataset {train_dataset.__class__.__name__} (length: {len(dataset)})')



    # optimizer
    opt_cls = get_attribute(config.optimizer)
    if config.optimize == 'torch.optim.SGD':
        opt_args = {'momentum': config.momentum if 'momentum' in config else 0}
    else:
        opt_args = {}
    opt = opt_cls(model.parameters(), lr=config.lr, **opt_args)

    batch_size, max_iterations = config.batch_size, config.max_iterations

    loss_fn = get_attribute(config.loss)

    if config.amp:
        log.info('Using AMP')
        autocast_fn = autocast
        scaler = GradScaler()
    else:
        autocast_fn, scaler = nullcontext, None

    save_only_trainable = True
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    # disable config when hyperparam. opt. to avoid writing logs.
    tracker_config = config if not config.hyperparameter_optimization else None

    with TrainingLogger(log_dir=config.name, model=model, config=tracker_config) as logger:

        i = 0
        while True:
            for data_x, data_y in data_loader:

                # between caption and output feature.
                # 1. Sample random captions
                # 2. Check alignment with CLIP

                # randomly mix text and visual support conditionals

                with autocast_fn():
                    # data_x[1] = text label
                    # prompts = model.sample_prompts(data_x[1])
                    prompts = data_x[1]
                    # model.clip_model()
                    text_cond = model.compute_conditional(prompts)
                    # OUR CASE
                    # data_x[2] = visual prompt
                    visual_s_cond, _, _ = model.visual_forward(data_x[2].cuda())


                batch_size = text_cond.shape[0]

                # sample weights for each element in batch
                text_weights = torch.distributions.Uniform(0, 1).sample((batch_size,))[:,
                               None]
                text_weights = text_weights.cuda()

                cond = text_cond * text_weights + visual_s_cond * (1 - text_weights)  # VECTOR DE CONDICIONAMIENTO


                with autocast_fn():
                    visual_q = None

                    pred, visual_q, _, _ = model(data_x[0].cuda(), cond, return_features=True)

                    loss = loss_fn(pred, data_y[0].cuda()) + dice_loss(pred, data_y[0].cuda())
                    writer.add_scalar('loss_train', loss, i)

                    if torch.isnan(loss) or torch.isinf(loss):
                        # skip if loss is nan
                        log.warning('Training stopped due to inf/nan loss.')
                        sys.exit(-1)

                    extra_loss = 0
                    loss += extra_loss

                opt.zero_grad()

                if scaler is None:
                    loss.backward()
                    opt.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()



                logger.iter(i=i, loss=loss)
                i += 1

                if i >= max_iterations:

                    if not isfile(join(logger.base_path, 'weights.pth')):
                        # only write if no weights were already written
                        logger.save_weights(only_trainable=save_only_trainable)

                    sys.exit(0)

                if config.checkpoint_iterations is not None and i in config.checkpoint_iterations:
                    logger.save_weights(only_trainable=save_only_trainable, weight_file='weights.pth')

                if val_interval is not None and i % val_interval == val_interval - 1:

                    val_loss, val_scores, maximize = validate(model, dataset_val, config)
                    writer.add_scalar('loss_val', val_loss, i)
                    writer.flush()
                    if len(val_scores) > 0:

                        score_str = f', scores: ' + ', '.join(f'{k}: {v}' for k, v in val_scores.items())

                        if maximize and val_scores[config.use_val_metric] > best_val_score:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_score = val_scores[config.use_val_metric]

                        elif not maximize and val_scores[config.use_val_metric] < best_val_score:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_score = val_scores[config.use_val_metric]

                    else:
                        score_str = ''
                        # if no score is used, fall back to loss
                        if val_loss < best_val_loss:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_loss = val_loss

                    log.info(f'Validation loss: {val_loss}' + score_str)
                    logger.iter(i=i, val_loss=val_loss, extra_loss=float(extra_loss), **val_scores)
                    model.train()

            print('epoch complete')


if __name__ == '__main__':
    main()