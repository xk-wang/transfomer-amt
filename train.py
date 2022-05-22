from distutils.command.config import config
import time
import torch
import torch.nn as nn
# import torch.optim as optim
# import fairseq.optim as optim
import torch_optimizer as optim
from datasets import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from configs import *
from transformer import Transformer
from easydict import EasyDict
from datetime import datetime
from evaluate import compute_metrics

if __name__ == "__main__":


    configs = EasyDict({'train_dir': '/home/data/wxk/master-graduate/maestro-v1.0.0-spits/train', 
                        'device': 'cuda:2',
                        'gpus': 1,
                        'batch_size': 8,
                        'logdir': 'transformer-' + datetime.now().strftime('%y%m%d-%H'),
                        'writer_interval': 100,
                        'log_interval': 50,
                        'save_interval': 10000,
                        'min_step': 10000})

    writer = SummaryWriter(configs.logdir)
    loader = DataLoader(MaestroDataset(configs), batch_size=configs.gpus*configs.batch_size, 
                        shuffle=False, collate_fn=pad_collate)#, num_workers=4)
    model = Transformer().to(configs.device)
    if configs.gpus>1:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)         # 忽略 占位符.
    step = 0
    write_graph = True
    optimizer = optim.Adafactor(model.parameters(), lr=1e-3)

    for epoch in range(400):
        for enc_inputs, dec_inputs, dec_outputs, enc_input_lens in loader:  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]
            start = time.time()           
            input_length_pad = INPUT_LENGTH
            inputs_mask = torch.stack([torch.cat([torch.ones((enc_input_lens[i], ), dtype=torch.int8), 
                                        torch.zeros((input_length_pad-enc_input_lens[i], ), dtype=torch.int8)])
                                        for i in range(len(enc_input_lens))])
            inputs_mask = inputs_mask.to(configs.device)
            enc_inputs = enc_inputs.to(configs.device)
            dec_inputs = dec_inputs.type(torch.LongTensor).to(configs.device)
            dec_outputs = dec_outputs.type(torch.LongTensor).to(configs.device)

            if write_graph:
                if configs.gpus>1:
                    writer.add_graph(model.module, (enc_inputs[:configs.batch_size, ], 
                                inputs_mask[:configs.batch_size, ], dec_inputs[configs.batch_size:, ]))
                else:
                    writer.add_graph(model, (enc_inputs, inputs_mask, dec_inputs))
                write_graph = False

            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, inputs_mask, dec_inputs)
            outputs = outputs.view(-1, TOTAL_EVENT_NUM)
            dec_outputs = dec_outputs.view(-1)
            loss = criterion(outputs, dec_outputs)
            metrics = compute_metrics(outputs, dec_outputs)
            metrics['loss'] = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step%configs.log_interval==0:
                print("====== training results: %.2fs ======"%(time.time()-start))
                print("step: %d loss: %-6.2f acc: %-4.2f%%"%(step, metrics["loss"], metrics["acc"]*100))
            

            if step%configs.writer_interval==0:
                writer.add_scalar("transformer/loss", metrics["loss"], global_step=step)
                writer.add_scalar("transformer/acc", metrics["acc"], global_step=step)
            
            if step>=configs.min_step and step%configs.save_interval ==0:
                print("====== saving checkpoints ======")
                if configs.gpus>1:  
                    torch.save(model.module.state_dict(), os.path.join(configs.logdir, f'model-{step}.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(configs.logdir, f'model-{step}.pt'))
                torch.save(optimizer.state_dict(), os.path.join(configs.logdir, 'last-optimizer-state.pt'))

    print("final loss: %-6.2f acc: %-4.2f%%"%(metrics["loss"], metrics["acc"]*100))
