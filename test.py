import os, sys, glob
import random
import time, datetime
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import re

def reset_model_parameters(model_name, test_options, save_subdir):
    if "cyclegan" in model_name:
        print("load cyclegan based model")
        base_model = "cyclegan"
    elif "cycada" in model_name:
        print("load cycada based model")
        base_model = "cycada"
    else:
        print("model not found")
        return None, 0, 1 #continue to search models
    opt = test_options.reset_model_options(base_model)
    with open(os.path.join("checkpoints", model_name, "train_opt.txt")) as f:
        lines = f.readlines()
    c_name = [re.match(r'c_.*="(.*)"|c_.*=(None)', line.strip()).group(1) for line in lines if line.startswith('c_A_name=') or line.startswith('c_B_name=')]
    if len(c_name)>1:
        if c_name[0] is not None:
            if not c_name[0].endswith(".pt"):
                c_name[0] += ".pt"
        if c_name[1] is not None:
            if not c_name[1].endswith(".pt"):
                c_name[1] += ".pt"
        opt.c_A_name=c_name[0]
        opt.c_B_name=c_name[1]
    opt.dataroot = [re.match(r'^dataroot="(.*)"', line.strip()).group(1) for line in lines if line.startswith('dataroot=')][0]
    opt.input_nc = int([re.match(r'^input_nc=([0-9]+)', line.strip()).group(1) for line in lines if line.startswith('input_nc=')][0])
    opt.output_nc = int([re.match(r'^output_nc=([0-9]+)', line.strip()).group(1) for line in lines if line.startswith('output_nc=')][0])
    niter = int([re.match(r'^niter=([0-9]+)', line.strip()).group(1) for line in lines if line.startswith('niter=')][0])
    niter_decay = int([re.match(r'^niter_decay=([0-9]+)', line.strip()).group(1) for line in lines if line.startswith('niter_decay=')][0])
    crop_size = int([re.match(r'^crop_size=([0-9]+)', line.strip()).group(1) for line in lines if line.startswith('crop_size=')][0])
    opt.direction = [re.match(r'^direction="(.*)"', line.strip()).group(1) for line in lines if line.startswith('direction=')][0]
    opt.crop_size=crop_size
    opt.load_size=crop_size #no crop
    opt.ngf = int([re.match(r'^ngf=([0-9]+)', line.strip()).group(1) for line in lines if line.startswith('ngf=')][0])
    opt.netG = [re.match(r'^netG="(.*)"', line.strip()).group(1) for line in lines if line.startswith('netG=')][0]
    netC = [re.match(r'^netC="(.*)"', line.strip()).group(1) for line in lines if line.startswith('netC=')][0]
    opt.norm = [re.match(r'^norm="(.*)"', line.strip()).group(1) for line in lines if line.startswith('norm=')][0]
    #ndf = [re.match(r'ndf=([0-9]+)', line.strip()).group(1) for line in lines if ('ndf' in line)]
    #opt.ndf = int(ndf[0])
    opt.model = base_model
    opt.name = model_name
    opt.save_subdir = save_subdir

    for i in range(1, 1000):
        if "%s_net_G_A.pth"%i in next(os.walk(os.path.join("checkpoints", model_name, "models")))[2]:
            pass
        else:
            total_epoch = i - 1 if i > 1 else 0
            break
    return opt, total_epoch, niter+niter_decay

if __name__ == '__main__':
    test_options = TestOptions()
    opt = test_options.parse(isPrint=False)  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1     # test code only supports num_threads = 1
    opt.serial_batches = 1  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = 1         # no flip; comment this line if results on flipped images are needed.

    # collect models
    check_dict = set([os.path.split(dir)[-1] for dir,_,_ in os.walk("checkpoints")])
    results_dict = set([os.path.split(dir)[-1] for dir,_,_ in os.walk("results")])
    model_dict = check_dict - (check_dict & results_dict) - set(["checkpoints", "images", "models"])
    if len(model_dict) == 0:
        print("no model to evaluate")
        exit(0)
    print("models should be evaluated\n", model_dict)
    print("remaining %d models"%len(model_dict))

    model_names = sorted(list(model_dict))
    #model_names = random.shuffle(list(model_dict))

    save_subdir = opt.name
    for model_name in model_names:
        print(model_name)
        opt, total_epoch, end_iter = reset_model_parameters(model_name, test_options, save_subdir)
        print("trained epoch", total_epoch, "end epoch", end_iter)
        if total_epoch < end_iter:
            print(model_name, "is not trained enough")
            continue
        test_options.print_options(opt)
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        batch_size = opt.batch_size
        print('data: %s, the number of test images = %d' %(opt.dataroot, dataset_size))
        results = {}
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        visualizer.print_message(datetime.datetime.now())
        visualizer.print_message("model: {}, total epoch: {}".format(model_name, total_epoch))

        for epoch in range(1, total_epoch + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            opt.epoch = epoch
            model = create_model(opt)      # create a model given opt.model and other options
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.set_device()
            # test with eval mode. This only affects layers like batchnorm and dropout.
            if opt.eval:
                model.eval()
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            for i, data in enumerate(dataset):
                iter_start_time = time.time()  # timer for computation per iteration
                t_data = iter_start_time - iter_data_time
                t_comp = (time.time() - iter_start_time) / batch_size
                visualizer.reset()
                epoch_iter += batch_size
                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference
                visualizer.display_current_results(model.get_current_visuals(), epoch, epoch_iter) #total_iters
                visualizer.set_scores(epoch, epoch_iter, float(epoch_iter) / dataset_size, model.get_current_scores(), t_comp, t_data)
            best_dict = visualizer.plot_current_scores(epoch)
            del model
        results[opt.name] = best_dict
        for key, value in results.items():
            visualizer.print_message("best result {}: {}".format(key, value))
        del visualizer
        break


