import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from model import Generator, Discriminator
from data import *
from itertools import chain
from scipy.misc import imsave

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 1000
CUDA = True
LR = 0.0002
UPDATE_STEP = 4
SAVE_EPOCH = 50
DECAYING_RATE = 0.95
DECAYING_STEP = 10
MODEL_DIR = './model'
RESULT_DIR = './result'

def GAN_loss(real, fake, cuda):
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    label_real = Variable(torch.ones([real.size()[0], 1]))
    label_fake = Variable(torch.zeros([fake.size()[0], 1]))
    label_gen_real = Variable(torch.ones([fake.size()[0], 1]))
    label_gen_fake = Variable(torch.zeros([real.size()[0], 1]))
    if cuda:
        label_real = label_real.cuda()
        label_fake = label_fake.cuda()
        label_gen_real = label_gen_real.cuda()
        label_gen_fake = label_gen_fake.cuda()
    dis_loss = criterion(real.view(-1, 1), label_real) + criterion(fake.view(-1, 1), label_fake)
    gen_loss = criterion(fake.view(-1, 1), label_gen_real) + criterion(real.view(-1, 1), label_gen_fake)
    return dis_loss, gen_loss


if __name__ == '__main__':
    A_tr, A_te, B_tr, B_te = data_read('./data', 'horse', 'zebra', IMAGE_SIZE, BATCH_SIZE, True)
    generatorA = Generator()
    generatorB = Generator()
    discriminatorA = Discriminator()
    discriminatorB = Discriminator()
    if CUDA:
        generatorA = generatorA.cuda()
        generatorB = generatorB.cuda()
        discriminatorA = discriminatorA.cuda()
        discriminatorB = discriminatorB.cuda()
    gen_params = chain(generatorA.parameters(), generatorB.parameters())
    dis_params = chain(discriminatorA.parameters(), discriminatorB.parameters())
    optim_gen = optim.Adam(gen_params, lr=LR, betas=(0.5, 0.999), weight_decay=1e-5)
    optim_dis = optim.Adam(dis_params, lr=LR, betas=(0.5, 0.999), weight_decay=1e-5)
    gen_scheduler = lr_scheduler.StepLR(optim_gen, step_size=DECAYING_STEP, gamma=DECAYING_RATE)
    dis_scheduler = lr_scheduler.StepLR(optim_dis, step_size=DECAYING_STEP, gamma=DECAYING_RATE)
    # MSE = nn.MSELoss()
    L1 = nn.L1Loss()
    for epoch in range(EPOCHS):
        if epoch >= 100:
            gen_scheduler.step()
            dis_scheduler.step()
        total_loss = [0, 0, 0, 0, 0, 0]
        for step, (A, B) in enumerate(zip(A_tr, B_tr)):
            generatorA.zero_grad()
            generatorB.zero_grad()
            discriminatorA.zero_grad()
            discriminatorB.zero_grad()
            A = Variable(A)
            B = Variable(B)
            if CUDA:
                A = A.cuda()
                B = B.cuda()
            AB = generatorB(A)
            BA = generatorA(B)
            ABA = generatorA(AB)
            BAB = generatorB(BA)
            # recon_loss_A = MSE(ABA, A)
            # recon_loss_B = MSE(BAB, B)
            recon_loss_A = L1(ABA, A)
            recon_loss_B = L1(BAB, B)
            dis_loss_A, gen_loss_A = GAN_loss(discriminatorA(A), discriminatorA(BA), CUDA)
            dis_loss_B, gen_loss_B = GAN_loss(discriminatorB(B), discriminatorB(AB), CUDA)
            gen_loss = (recon_loss_A + recon_loss_B) * 10 + (gen_loss_A + gen_loss_B)
            dis_loss = dis_loss_A + dis_loss_B
            if step % UPDATE_STEP == UPDATE_STEP - 1:
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()
            total_loss[0] += gen_loss_A.mean().data[0]
            total_loss[1] += gen_loss_B.mean().data[0]
            total_loss[2] += recon_loss_A.mean().data[0]
            total_loss[3] += recon_loss_B.mean().data[0]
            total_loss[4] += dis_loss_A.data[0]
            total_loss[5] += dis_loss_B.data[0]
        print('epoch %d log' % (epoch + 1))
        print('generator A GAN loss: %f' % total_loss[0])
        print('generator B GAN loss: %f' % total_loss[1])
        print('A reconstruction loss: %f' % total_loss[2])
        print('B reconstruction loss: %f' % total_loss[3])
        print('A discriminator loss: %f' % total_loss[4])
        print('B discriminator loss: %f' % total_loss[5])
        print('------------------------------')
        if epoch % SAVE_EPOCH == SAVE_EPOCH - 1:
            print('images and model saving at epoch %d' % (epoch + 1))
            idx_A = 0
            idx_B = 0
            os.mkdir(os.path.join(RESULT_DIR, str(epoch + 1)))
            for step, A in enumerate(A_te):
                A = Variable(A)
                if CUDA:
                    A = A.cuda()
                AB = generatorB(A)
                ABA = generatorA(AB)
                AB_image = (AB.cpu().data.numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
                ABA_image = (ABA.cpu().data.numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
                AB_image = [np.squeeze(x) for x in np.split(AB_image, AB_image.shape[0], 0)]
                ABA_image = [np.squeeze(x) for x in np.split(ABA_image, ABA_image.shape[0], 0)]
                for i in range(len(AB_image)):
                    imsave(os.path.join(os.path.join(RESULT_DIR, str(epoch + 1)), str(idx_A)+'.AB.jpg'), AB_image[i])
                    imsave(os.path.join(os.path.join(RESULT_DIR, str(epoch + 1)), str(idx_A) + '.ABA.jpg'), ABA_image[i])
                    idx_A += 1
            for step, B in enumerate(B_te):
                B = Variable(B)
                if CUDA:
                    B = B.cuda()
                BA = generatorA(B)
                BAB = generatorB(BA)
                BA_image = (BA.cpu().data.numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
                BAB_image = (BAB.cpu().data.numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
                BA_image = [np.squeeze(x) for x in np.split(BA_image, BA_image.shape[0], 0)]
                BAB_image = [np.squeeze(x) for x in np.split(BAB_image, BAB_image.shape[0], 0)]
                for i in range(len(BA_image)):
                    imsave(os.path.join(os.path.join(RESULT_DIR, str(epoch + 1)), str(idx_B) + '.BA.jpg'), BA_image[i])
                    imsave(os.path.join(os.path.join(RESULT_DIR, str(epoch + 1)), str(idx_B) + '.BAB.jpg'), BAB_image[i])
                    idx_B += 1
            torch.save(generatorA, os.path.join(MODEL_DIR, 'generatorA_%d' % (epoch + 1)))
            torch.save(generatorB, os.path.join(MODEL_DIR, 'generatorA_%d' % (epoch + 1)))
            torch.save(discriminatorA, os.path.join(MODEL_DIR, 'discriminatorA_%d' % (epoch + 1)))
            torch.save(discriminatorB, os.path.join(MODEL_DIR, 'discriminatorB_%d' % (epoch + 1)))
            print('images and model saved')
            print('------------------------------')
