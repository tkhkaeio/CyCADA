import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import ntpath
import time
from PIL import Image
from . import util
from subprocess import Popen, PIPE
try:
    from scipy.misc import imresize
except:
    from skimage.transform import resize as imresize

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        - Cache the training/test options
        - Create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        if self.opt.phase=="train":
            self.check_dir = os.path.join(opt.checkpoints_dir, opt.name)
        elif self.opt.phase=="test":
            self.check_dir = os.path.join(opt.results_dir, opt.save_subdir ,opt.name)
        self.img_dir = os.path.join(self.check_dir, 'images')
        util.mkdirs([self.check_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(self.check_dir, 'loss_log.txt')
        self.plot_data_scores = None
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch, iters):
        """Display current results; save current results to chechpoints/images

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
        """
        if not self.saved:
            self.saved = True
            vis_num = len(visuals.keys())
            if vis_num < 10:
                x = 2
                y = vis_num//2
            else:
                x = 4
                y = vis_num//4
            fig, axarr = plt.subplots(x,y)
            for ax, (label, (image, message)) in zip(axarr.ravel(), visuals.items()):
                image_numpy = util.tensor2im(image)
                image_pil = Image.fromarray(image_numpy)
                ax.imshow(image_pil)
                ax.set_axis_off()
                if self.opt.dataset_mode.startswith("class"):
                    if vis_num > 10:
                        ax.set_title(label + " " + message, fontsize=4)
                    elif vis_num >= 8:
                        ax.set_title(label + " " + message, fontsize=6)
                    else:
                        ax.set_title(label + " " + message, fontsize=10)
                else:
                    ax.set_title(label)
            fig.savefig(os.path.join(self.img_dir,"epoch%.3d_iter%.3d.png"%(epoch, iters)), dpi=300)
            plt.close('all')

    def plot_current_losses(self, epoch, epoch_iter, counter_ratio, losses, t_comp, t_data):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data_losses'):
            self.plot_data_losses = {'X': [], 'Y': [], 'legend': sorted(list(losses.keys()))}
        self.plot_data_losses['X'].append(epoch + counter_ratio)
        self.plot_data_losses['Y'].append([losses[k] for k in self.plot_data_losses['legend']])
        Y = np.asarray(self.plot_data_losses['Y'])
        unique_initial_set = (lambda initial_list: sorted(set(initial_list), key=initial_list.index))([l[0] for l in self.plot_data_losses['legend']])
        keys_nest = [[l for l in self.plot_data_losses['legend'] if i==l[0]] for i in unique_initial_set]
        num = -1
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("losses")
        for _, keys in enumerate(keys_nest):
            for k in keys:
                num += 1
                plt.plot(self.plot_data_losses['X'],Y.T[num], label=k)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.check_dir,"losses.png"), dpi=300)
        plt.close('all')

        num = -1
        for i, keys in enumerate(keys_nest):
            plt.figure()
            plt.xlabel("epoch")
            plt.ylabel("losses%d"%i)
            for k in keys:
                num += 1
                plt.plot(self.plot_data_losses['X'], Y.T[num], label=k)
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(self.check_dir,"losses%d.png"%i), dpi=300)
            plt.close('all')
        self.print_current_results(epoch, epoch_iter, losses, t_comp, t_data, "loss")

    def set_scores(self, epoch, epoch_iter, counter_ratio, scores, t_comp, t_data):
        if self.plot_data_scores is None: #hasattr(self, 'plot_data_scores'):
            self.plot_data_scores = {epoch: []}
            self.plot_data_counts = {epoch: []}
            self.plot_data_dists = {epoch: []}
            count_set = set([n for n in list(scores.keys()) if "count" in n])
            dist_set = set([n for n in list(scores.keys()) if "dist" in n])
            score_set = set(scores.keys()) - count_set - dist_set
            self.score_legend = sorted(list(score_set))
            self.count_legend = sorted(list(count_set))
            self.dist_legend = sorted(list(dist_set))
        elif not epoch in self.plot_data_scores.keys():
            self.plot_data_scores[epoch] = []
            self.plot_data_counts[epoch] = []
            self.plot_data_dists[epoch] = []
        self.plot_data_scores[epoch].append([scores[k] for k in self.score_legend])
        self.plot_data_counts[epoch].append([scores[k] for k in self.count_legend])
        self.plot_data_dists[epoch].append([scores[k] for k in self.dist_legend])
        #self.print_current_results(epoch, epoch_iter, scores, t_comp, t_data, "score")

    def plot_current_scores(self, epoch):
        """display the current scores on visdom display: dictionary of score labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            score (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        for epoch in self.plot_data_scores.keys():
            if len(self.plot_data_scores[epoch]) > 2: # aggregate
                result_score = np.mean(self.plot_data_scores[epoch], axis=0).tolist()
                result_count = np.sum(self.plot_data_counts[epoch], axis=0).tolist()
                result_dist = np.mean(self.plot_data_dists[epoch], axis=0).tolist()
                self.plot_data_scores[epoch] = [result_score]
                self.plot_data_counts[epoch] = [result_count]
                self.plot_data_dists[epoch] = [result_dist]
                #self.plot_data_scores[epoch].append(result_score)
        if epoch > 1:
            X = list(map(int, self.plot_data_scores.keys()))
            Y = np.squeeze(np.asarray(list(self.plot_data_scores.values())), 1)
            n_fig = 4 if Y.shape[0]>10 else 2
            best_result = ("-"*30)+"\n"
            best_dict = {}
            for i, k in enumerate(self.score_legend):
                if i%n_fig == 0:
                    plt.figure()
                    plt.xlabel("epoch")
                    plt.ylabel("score")
                best_score = Y.T[i].max()
                plt.plot(X, Y.T[i], label="{}, best {:.4f} ".format(k, best_score))
                best_result += "best {}: {:.4f} ".format(k, best_score)
                best_dict[k] = best_score
                if i%n_fig == n_fig-1:
                    plt.legend(loc='lower left')
                    plt.savefig(os.path.join(self.check_dir,"scores_%i.png"%(i//n_fig)), dpi=300)
                    plt.close('all')
            best_result += ("\n"+("-"*30))

            plt.figure()
            plt.xlabel("epoch")
            plt.ylabel("score")
            for i, k in enumerate(self.score_legend):
                plt.plot(X, Y.T[i], label="{}, best {:.4f} ".format(k, Y.T[i].max()))
            plt.legend(loc='lower left')
            plt.savefig(os.path.join(self.check_dir,"scores.png"), dpi=300)
            plt.close('all')
            self.print_message(best_result)

            # print((self.plot_data_counts.keys()))
            # print((self.plot_data_counts.values()))
            # print(len(list(self.plot_data_counts.values())[0]))
            # print(len(list(self.plot_data_counts.values())[0][0]))
            if len(list(self.plot_data_counts.values())[0][0])>1:
                X = list(map(int, self.plot_data_counts.keys()))
                Y = np.squeeze(np.asarray(list(self.plot_data_counts.values())), 1)
                plt.figure()
                plt.xlabel("epoch")
                plt.ylabel("count")
                for i, k in enumerate(self.count_legend):
                    mean_score = Y.T[i].mean()
                    plt.plot(X, Y.T[i], label="{}, mean {:.4f} ".format(k, mean_score))
                plt.legend(loc='lower left')
                plt.savefig(os.path.join(self.check_dir,"count.png"), dpi=300)
                plt.close('all')

            if len(list(self.plot_data_dists.values())[0][0])>1:
                X = list(map(int, self.plot_data_dists.keys()))
                Y = np.squeeze(np.asarray(list(self.plot_data_dists.values())), 1)
                plt.figure()
                plt.xlabel("epoch")
                plt.ylabel("dist")
                for i, k in enumerate(self.dist_legend):
                    if not "std" in k:
                        mean_score = Y.T[i].mean()
                        std_score = Y.T[i].std(ddof=1)
                        plt.plot(X, Y.T[i], label="{}, mean {:.4f} std: {:.4f}".format(k, mean_score, std_score))
                plt.legend(loc='lower left')
                plt.savefig(os.path.join(self.check_dir,"dist.png"), dpi=300)
                plt.close('all')

                plt.figure()
                plt.xlabel("epoch")
                plt.ylabel("dist_std")
                for i, k in enumerate(self.dist_legend):
                    if "std" in k:
                        mean_score = Y.T[i].mean()
                        plt.plot(X, Y.T[i], label="{}, mean {:.4f} ".format(k, mean_score))
                plt.legend(loc='lower left')
                plt.savefig(os.path.join(self.check_dir,"dist_std.png"), dpi=300)
                plt.close('all')
            return best_dict

    def plot_current_preds(self, epoch, epoch_iter, counter_ratio, preds, t_comp, t_data):
        """display the current predictions on visdom display: dictionary of pred labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            preds (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data_preds'):
            self.plot_data_preds = {'X': [], 'Y': [], 'legend': sorted(list(preds.keys()))}
        self.plot_data_preds['X'].append(epoch + counter_ratio)
        self.plot_data_preds['Y'].append([preds[k] for k in self.plot_data_preds['legend']])
        Y = np.asarray(self.plot_data_preds['Y'])
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("preds")
        plt.ylim((-0.1,1.1))
        for i,k in enumerate(self.plot_data_preds['legend']):
            plt.plot(self.plot_data_preds['X'],Y.T[i], label=k)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.check_dir, "preds.png"), dpi=300)
        plt.close('all')
        self.print_current_results(epoch, epoch_iter, preds, t_comp, t_data, "pred")

    def plot_current_gnorms(self, epoch, epoch_iter, counter_ratio, gnorms, t_comp, t_data):
        """display the current gradient norms on visdom display: dictionary of gnorm labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            gnorms (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data_gnorms'):
            self.plot_data_gnorms = {'X': [], 'Y': [], 'legend': sorted(list(gnorms.keys()))}
        self.plot_data_gnorms['X'].append(epoch + counter_ratio)
        self.plot_data_gnorms['Y'].append([gnorms[k] for k in self.plot_data_gnorms['legend']])
        Y = np.asarray(self.plot_data_gnorms['Y'])
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("gnorms")
        for i,k in enumerate(self.plot_data_gnorms['legend']):
            plt.plot(self.plot_data_gnorms['X'],Y.T[i], label=k)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.check_dir, "gnorms.png"), dpi=300)
        plt.close('all')
        self.print_current_results(epoch, epoch_iter, gnorms, t_comp, t_data, "gnorm")

    # results: are losses, scpres, preds, gnorms
    def print_current_results(self, epoch, iters, results, t_comp, t_data, name=None):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            results (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) [%5s] ' % (epoch, iters, t_comp, t_data, name)
        for k, v in results.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    # results: are losses, scpres, preds, gnorms
    def print_message(self, message):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            results (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message