#!/usr/bin/python3

from absl import flags, app
import torch
from torch import device, save, load, autograd
from torch.nn import L1Loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import distributed
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from create_datasets import Molecule
from models import ConductivityPredictor

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_csv', default = None, help = 'path to input csv')
  flags.DEFINE_integer('batch', default = 32, help = 'batch size')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_integer('decay_steps', default = 2000, help = 'decay steps')
  flags.DEFINE_integer('workers', default = 16, help = 'number of workers')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  trainset = Molecule(FLAGS.input_csv)
  dist.init_process_group(backend = 'nccl')
  torch.cuda.set_device(dist.get_rank())
  trainset_sampler = distributed.DistributedSampler(trainset)
  if dist.get_rank() == 0:
    print(f'trainset size: {len(trainset)}')
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch, shuffle = False, num_workers = FLAGS.workers, sampler = trainset_sampler, pin_memory = False)
  model = ConductivityPredictor()
  model.to(device(FLAGS.device))
  model = DDP(model, device_ids = [dist.get_rank()], output_device = dist.get_rank(), find_unused_parameters = True)
  mae = L1Loss()
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  if dist.get_rank() == 0:
    if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
    tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  start_epoch = 0
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.pth'))
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    start_epoch = ckpt['epoch']
  for epoch in range(start_epoch, FLAGS.epochs):
    train_dataloader.sampler.set_epoch(epoch)
    model.train()
    for step, data in enumerate(train_dataloader):
      optimizer.zero_grad()
      data = data.to(device(FLAGS.device))
      c = model(data)
      loss = mae(data.y, c)
      loss.backward()
      optimizer.step()
      global_steps = epoch * len(train_dataloader) + step
      if global_steps % 100 == 0 and dist.get_rank() == 0:
        print(f'Step #{global_steps} Epoch #{epoch}: loss {loss}')
        tb_writer.add_scalar('loss', loss, global_steps)
    scheduler.step()
    if dist.get_rank() == 0:
      ckpt = {'epoch': epoch,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler}
      save(ckpt, join(FLAGS.ckpt, f'model-ep{epoch}.pth'))

if __name__ == "__main__":
  add_options()
  app.run(main)

