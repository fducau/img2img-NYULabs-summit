import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import networks


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt['gpu_ids']
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = opt['outf']

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        pass


class netModel(BaseModel):
    def name(self):
        return 'netModel'

    def initialize(self, opt, train_mode=True):
        # Model transforms from A --> B and uses Adv as the
        # adversarial example.
        BaseModel.initialize(self, opt)
        self.train_mode = train_mode
        # define tensors
        self.input_B = self.Tensor(opt['batchSize'], opt['input_nc'],
                                   opt['B_height'], opt['B_width'])
        self.input_A = self.Tensor(opt['batchSize'], opt['output_nc'],
                                   opt['A_height'], opt['A_width'])

        # load/define networks
        self.netG = networks.define_G(opt['input_nc'], opt['output_nc'], opt['ngf'],
                                      opt['norm'], self.gpu_ids)

        if self.train_mode:
            use_sigmoid = opt['no_lsgan']
            self.netD = networks.define_D(opt['input_nc'] + opt['output_nc'], opt['ndf'],
                                          opt['which_model_netD'],
                                          opt['n_layers_D'], use_sigmoid, self.gpu_ids)

        if self.train_mode:
            # self.fake_AB_pool = ImagePool(opt['pool_size'])
            self.old_lr = opt['lr']
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt['no_lsgan'], tensor=self.Tensor)
            self.content_loss = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt['lr'], betas=(opt['beta1'], 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt['lr'], betas=(opt['beta1'], 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        if self.train_mode:
            input_B = input[0][0]
            input_A = input[1][0]

            self.input_B.resize_(input_B.size()).copy_(input_B)
            self.input_A.resize_(input_A.size()).copy_(input_A)

        else:
            input_A = input[0][0]
            self.input_A.resize_(input_A.size()).copy_(input_A)

    def forward(self):
        if self.train_mode:
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG.forward(self.real_A)
            self.real_B = Variable(self.input_B)
        else:
            # Do not backprop gradients
            self.real_A = Variable(self.input_A, volatile=True)
            self.fake_B = self.netG.forward(self.real_A)

    def backward_D(self):
        # stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.pred_fake = self.netD.forward(fake_AB.detach())

        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        self.loss_D_fake.backward()

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real.backward()
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_content = self.content_loss(self.fake_B, self.real_B)
        self.loss_G = self.loss_G_content + self.loss_G_GAN * self.opt['L1lambda']

        self.loss_G.backward()

    def optimize_parameters(self):
        '''
        Run forward and backward pathds and apply optimization step
        '''
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.train_mode:
            return OrderedDict([
                ('G_GAN', self.loss_G_GAN.data[0]),
                ('G_L1', self.loss_G_content.data[0]),
                ('D_real', self.loss_D_real.data[0]),
                ('D_fake', self.loss_D_fake.data[0]),
            ])

        raise UnboundLocalError('Errors are only computed in when train_mode is True')

    def get_current_visuals(self, test=False):
        # fake_in = util.tensor2im(self.fake_in.data)
        # fake_out = util.tensor2im(self.fake_out.data)
        # real_out = util.tensor2im(self.real_out.data)
        if test or not self.train_mode:
            return OrderedDict('fake_out', self.fake_B)

        return OrderedDict([('fake_in', self.real_A),
                            ('fake_out', self.fake_B),
                            ('real_out', self.real_B)])

    def update_learning_rate(self):
        lr = self.old_lr / 10.
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
