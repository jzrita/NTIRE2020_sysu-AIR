import argparse, random
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset


def main():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.', default='options/train/train_SRFBN.json')
    opt = option.parse(parser.parse_args().opt)
    writer = SummaryWriter(comment=f'wjh')

    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)
    print("===> Random Seed: [%d]"%seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('===> Train Dataset: %s   Number of images: [%d]' % (train_set.name(), len(train_set)))
            if train_loader is None: raise ValueError("[Error] The training data does not exist")

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val_set.name(), len(val_set)))

        else:
            raise NotImplementedError("[Error] Dataset phase [%s] in *.json is not recognized." % phase)

    solver = create_solver(opt)

    scale = opt['scale']
    model_name = opt['networks']['which_model'].upper()

    print('===> Start Train')
    print("==================================================")

    solver_log = solver.get_current_log()

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']

    print("Method: %s || Scale: %d || Epoch Range: (%d ~ %d)"%(model_name, scale, start_epoch, NUM_EPOCH))

    step = 0
    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print('\n===> Training Epoch: [%d/%d]...  Learning Rate: %f'%(epoch,
                                                                      NUM_EPOCH,
                                                                      solver.get_current_learning_rate()))

        # Initialization
        solver_log['epoch'] = epoch

        # Train model
        train_loss_list = []
        with tqdm(total=len(train_loader), desc='Epoch: [%d/%d]'%(epoch, NUM_EPOCH), miniters=1) as t:
            for iter, batch in enumerate(train_loader):
                step += 1
                solver.feed_data(batch)
                iter_loss, loss_log = solver.train_step(step)
                batch_size = batch['LR'].size(0)
                train_loss_list.append(iter_loss*batch_size)
                t.set_postfix_str(
                    '''Batch Loss: {}, pixel_loss: {}, feature_loss: {}, tv_loss: {}, style_loss: {}, 
                    fft_loss: {}, 
                    generator_vanilla_loss: {}, discriminator_loss: {}'''.format(
                        iter_loss, loss_log["pixel_loss"], loss_log["feature_loss"], loss_log["tv_loss"], loss_log["style_loss"],
                        loss_log['fft_loss'],
                        loss_log['generator_vanilla_loss'], loss_log['discriminator_loss']
                ))

                writer.add_scalar('Batch Loss/train', iter_loss, step)
                writer.add_scalar('pixel_loss/train', loss_log["pixel_loss"], step)
                writer.add_scalar('feature_loss/train', loss_log["feature_loss"], step)
                writer.add_scalar('tv_loss/train', loss_log["tv_loss"], step)
                writer.add_scalar('style_loss/train', loss_log["style_loss"], step)
                writer.add_scalar('fft_loss/train', loss_log["fft_loss"], step)
                writer.add_scalar('generator_vanilla_loss/train', loss_log["generator_vanilla_loss"], step)
                writer.add_scalar('discriminator_loss/train', loss_log["discriminator_loss"], step)

                t.update()

        solver_log['records']['train_loss'].append(sum(train_loss_list)/len(train_set))
        solver_log['records']['lr'].append(solver.get_current_learning_rate())

        print('\nEpoch: [%d/%d]   Avg Train Loss: %.6f' % (epoch,
                                                    NUM_EPOCH,
                                                    sum(train_loss_list)/len(train_set)))

        print('===> Validating...',)

        psnr_list = []
        ssim_list = []
        val_loss_list = []

        for iter, batch in enumerate(val_loader):
            solver.feed_data(batch)
            iter_loss = solver.test()
            val_loss_list.append(iter_loss)

            # calculate evaluation metrics
            visuals = solver.get_current_visual()
            psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if opt["save_image"]:
                solver.save_current_visual(epoch, iter)

        solver_log['records']['val_loss'].append(sum(val_loss_list)/len(val_loss_list))
        solver_log['records']['psnr'].append(sum(psnr_list)/len(psnr_list))
        solver_log['records']['ssim'].append(sum(ssim_list)/len(ssim_list))

        writer.add_scalar('val_loss/val', sum(val_loss_list)/len(val_loss_list), step)
        writer.add_scalar('psnr/val', sum(psnr_list)/len(psnr_list), step)
        writer.add_scalar('ssim/val', sum(ssim_list)/len(ssim_list), step)

        # record the best epoch
        epoch_is_best = False
        if solver_log['best_pred'] < (sum(psnr_list)/len(psnr_list)):
            solver_log['best_pred'] = (sum(psnr_list)/len(psnr_list))
            epoch_is_best = True
            solver_log['best_epoch'] = epoch

        print("[%s] PSNR: %.2f   SSIM: %.4f   Loss: %.6f   Best PSNR: %.2f in Epoch: [%d]" % (val_set.name(),
                                                                                              sum(psnr_list)/len(psnr_list),
                                                                                              sum(ssim_list)/len(ssim_list),
                                                                                              sum(val_loss_list)/len(val_loss_list),
                                                                                              solver_log['best_pred'],
                                                                                              solver_log['best_epoch']))

        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best)
        solver.save_current_log()

        # update lr
        solver.update_learning_rate(epoch)
    writer.close()
    print('===> Finished !')


if __name__ == '__main__':
    main()