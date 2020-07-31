import os, sys
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import numpy as np

class CyCADAModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_sem_A', type=float, default=1.0, help='weight for semantic loss (A -> B -> A)')
            parser.add_argument('--lambda_sem_B', type=float, default=1.0, help='weight for semantic loss (B -> A -> B)')
            parser.add_argument('--c_lr', type=float, default=0.0001, help='learning rate of classifier')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.score_names = ['acc_real_A', 'acc_fake_B', 'acc_rec_A', 'acc_real_B', 'acc_fake_A', 'acc_rec_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        if self.isTrain:
            # specify the training losses you want to print out. The program will call base_model.get_current_losses
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'sem_B', 'sem_A', 'C_B_task', 'C_B_adv', 'C_B', 'D_ft_adv']
            self.score_names.extend(['acc_D_ft_source', 'acc_D_ft_target'])
            # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'C_B', 'D_ft']
            if opt.pretrain:
                self.loss_names.append('C_A')
                self.model_names.append("C_A")
            if self.opt.monitor_gnorm:
                self.gnorm_names = self.model_names
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'C_B', 'D_ft']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netC_A = networks.define_C(opt.output_nc, opt.netC, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netC_B = networks.define_C(opt.output_nc, opt.netC, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_ft = networks.define_C(opt.output_nc, "d_ft", opt.init_type, opt.init_gain, self.gpu_ids)
        if not opt.pretrain:
            print("load a pretrain model")
            if isinstance(self.netC_A, torch.nn.DataParallel):
                self.netC_A = self.netC_A.module
            self.netC_A.load_state_dict(torch.load(os.path.join("models/pretrain", opt.c_A_name), map_location=self.map_location))
            self.netC_A.eval()
        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCLS = torch.nn.modules.CrossEntropyLoss()
            self.Batch_MSE = lambda a, b: torch.nn.MSELoss(reduction='none')(a.view(a.size()[0], -1),b.view(b.size()[0], -1)).mean(dim=1)
            self.Batch_L1 = lambda a, b: torch.nn.L1Loss(reduction='none')(a.view(a.size()[0], -1),b.view(b.size()[0], -1)).mean(dim=1)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr/2., betas=(opt.beta1, 0.999))
            self.optimizer_C_A = torch.optim.Adam(self.netC_A.parameters(), lr=opt.c_lr, betas=(opt.beta1, 0.999))
            self.optimizer_C_B = torch.optim.Adam(self.netC_B.parameters(), lr=opt.c_lr/10, betas=(opt.beta1, 0.999))
            self.optimizer_D_ft = torch.optim.Adam(self.netD_ft.parameters(), lr=opt.c_lr/10, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_pretrain_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.target_A = input['A_target' if AtoB else 'B_target'].to(self.device)

    def backward_C_A(self):
        # forward only real source image through semantic classifier
        self.loss_C_A = self.criterionCLS(self.netC_A(self.real_A), self.target_A) # C_A(A)
        self.loss_C_A.backward()

    def pretrain(self):
        # forward
        self.pred_real_A = self.netC_A(self.real_A)
        label_real_A, score_acc_real_A = networks.prediction(self.pred_real_A, self.target_A)
        # C_A
        self.set_requires_grad([self.netC_A], True)
        self.optimizer_C_A.zero_grad()
        self.backward_C_A()
        self.optimizer_C_A.step()
        return score_acc_real_A, self.loss_C_A


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.target_A = input['A_target' if AtoB else 'B_target'].to(self.device)
        self.target_B = input['B_target' if AtoB else 'A_target'].to(self.device)
        #use for feature discriminator
        self.source_label = torch.ones(self.real_A.size()[0]).long().to(self.device)
        self.target_label = torch.zeros(self.real_B.size()[0]).long().to(self.device)

    def forward(self):
        self.fake_A = self.netG_B(self.real_B) # G_B(B)
        self.fake_B = self.netG_A(self.real_A) # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

        self.pred_real_A = self.netC_A(self.real_A) # C_A(real_A)
        self.label_real_A, self.score_acc_real_A = networks.prediction(self.pred_real_A, self.target_A)
        self.pred_real_B = self.netC_B(self.real_B) # C_B(real_B)
        self.label_real_B, self.score_acc_real_B = networks.prediction(self.pred_real_B, self.target_B)
        self.pred_fake_A = self.netC_A(self.fake_A) # C_A(fakel_A)
        self.label_fake_A, self.score_acc_fake_A = networks.prediction(self.pred_fake_A, self.target_B)
        self.pred_fake_B = self.netC_B(self.fake_B) # C_B(fakel_B)
        self.label_fake_B, self.score_acc_fake_B = networks.prediction(self.pred_fake_B, self.target_A)
        self.pred_rec_A = self.netC_A(self.rec_A)   # C_A(rec_A)
        self.label_rec_A, self.score_acc_rec_A = networks.prediction(self.pred_rec_A, self.target_A)
        self.pred_rec_B  = self.netC_B(self.rec_B)  # C_B(rec_B)
        self.label_rec_B, self.score_acc_rec_B = networks.prediction(self.pred_rec_B, self.target_B)

        pred_source = self.netD_ft(self.netC_A(self.fake_B.detach())) # D_ft(C_A(G_A(A)))
        pred_target = self.netD_ft(self.netC_B(self.real_B))          # D_ft(C_B(B))
        _, self.score_acc_D_ft_source = networks.prediction(pred_source, self.source_label)
        _, self.score_acc_D_ft_target = networks.prediction(pred_target, self.target_label)
        self.score_acc_D_ft = (self.score_acc_D_ft_source + self.score_acc_D_ft_target)/2 #feature discriminator accuracy

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        if self.opt.gan_mode == "wgangp":
            gradient_penalty, gradients = networks.cal_gradient_penalty(netD,real,fake,self.device)
            gradient_penalty.backward(retain_graph=True)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        else:
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # Calculate discriminator accuracy
        true_labels = torch.ones(real.size()[0]).long()
        fake_labels = torch.zeros(fake.detach().size()[0]).long()
        _, true_acc = networks.prediction(pred_real.squeeze().cpu(), true_labels, onehot=False)
        _, fake_acc = networks.prediction(pred_fake.squeeze().cpu(), fake_labels, onehot=False)
        acc = (true_acc + fake_acc) * 0.5
        return loss_D, acc

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_basic_A, self.score_acc_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        self.loss_D_A = self.loss_D_basic_A
        self.loss_D_A.backward()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_basic_B, self.score_acc_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
        self.loss_D_B = self.loss_D_basic_B
        self.loss_D_B.backward()

    def backward_G(self):
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        # combined loss standard cyclegan
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        # semantic loss A
        self.loss_sem_A = self.criterionCLS(self.netC_A(self.fake_A), networks.prediction(self.netC_A(self.real_B))) * self.opt.lambda_sem_A
        # semantic loss B
        self.loss_sem_B = self.criterionCLS(self.netC_A(self.fake_B), networks.prediction(self.netC_A(self.real_A))) * self.opt.lambda_sem_B
        self.loss_G += self.loss_sem_A + self.loss_sem_B
        self.loss_G.backward()

    def backward_D_ft(self):
        # Source
        pred_source = self.netD_ft(self.netC_A(self.fake_B.detach()))
        loss_D_ft_s = self.criterionCLS(pred_source, self.source_label)
        # Target
        pred_target = self.netD_ft(self.netC_B(self.real_B))
        loss_D_ft_t = self.criterionCLS(pred_target, self.target_label)
        # Combined loss
        self.loss_D_ft_adv = (loss_D_ft_s + loss_D_ft_t) * 0.5
        self.loss_D_ft_adv.backward()

    def backward_C_B(self):
        pred_target = self.netC_B(self.fake_B.detach()) # C_B(G_A(A))
        self.loss_C_B_task = self.criterionCLS(pred_target, self.target_A)
        # only update net if discriminator is strong
        if self.score_acc_D_ft > 0.6:
            pred_target = self.netD_ft(self.netC_B(self.real_B)) # D_ft(C_B(B))
            target_label = torch.ones(self.real_B.size()[0]).long().to(self.device)
            self.loss_C_B_adv = self.criterionCLS(pred_target, target_label)
        else:
            self.loss_C_B_adv = 0
        self.loss_C_B = self.loss_C_B_task + self.loss_C_B_adv
        self.loss_C_B.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # C_A
        if self.opt.pretrain:
            self.set_requires_grad([self.netC_A], True)
            self.optimizer_C_A.zero_grad()
            self.backward_C_A()
            self.optimizer_C_A.step()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netC_A], False)
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.optimizer_G.zero_grad()
        self.optimizer_C_A.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
        # C_B
        self.set_requires_grad([self.netD_ft], False)
        self.set_requires_grad([self.netC_B], True)
        self.optimizer_C_B.zero_grad()
        self.backward_C_B()
        self.optimizer_C_B.step()
        # D_ft
        self.set_requires_grad([self.netC_B], False)
        self.set_requires_grad([self.netD_ft], True)
        self.optimizer_D_ft.zero_grad()
        self.backward_D_ft()
        self.optimizer_D_ft.step()
        # clip gradient of G, D, C
        if self.opt.monitor_gnorm:
            self.gnorm_G_A = torch.nn.utils.clip_grad_norm_(self.netG_A.parameters(), self.opt.max_gnorm)
            self.gnorm_G_B = torch.nn.utils.clip_grad_norm_(self.netG_B.parameters(), self.opt.max_gnorm)
            self.gnorm_D_A = torch.nn.utils.clip_grad_norm_(self.netD_A.parameters(), self.opt.max_gnorm)
            self.gnorm_D_B = torch.nn.utils.clip_grad_norm_(self.netD_B.parameters(), self.opt.max_gnorm)
            self.gnorm_C_A = torch.nn.utils.clip_grad_norm_(self.netC_A.parameters(), self.opt.max_gnorm)
            self.gnorm_C_B = torch.nn.utils.clip_grad_norm_(self.netC_B.parameters(), self.opt.max_gnorm)
            self.gnorm_D_ft = torch.nn.utils.clip_grad_norm_(self.netD_ft.parameters(), self.opt.max_gnorm)
