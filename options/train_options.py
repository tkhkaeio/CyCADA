from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        #parser.add_argument('--isTrain', default=True, help='whether to train(always true)') #bool
        # visualization parameters
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console in # of total interation')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--eval_step_freq', type=int, default=1000, help='frequency of evaluating checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', type=int, default=0, choices=[0,1], help='whether saves model by iteration')
        parser.add_argument('--continue_train', type=int, default=0, choices=[0,1], help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--pretrain_epoch', type=int, default=10, help='# of epoch in pretrain of classifier')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp | wgan]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--monitor_gnorm', type=int, default=0, choices=[0,1], help='flag set to monitor grad norms')
        parser.add_argument('--max_gnorm', type=float, default=500., help='max grad norm to which it will be clipped (if exceeded)')

        self.isTrain = True
        return parser
