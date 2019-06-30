import torch
import logging
import time
import datetime
from sklearn.metrics import accuracy_score
import numpy as np

from tools.checkpoint import Checkpointer

def do_train(name, model, device, trndata_loader, valdata_loader, optimizer, criterion, scheduler, nepochs, checkpoint_period, checkpointer):
    logger = logging.getLogger(name=name)
    logger.info("Start training")
    
    total_step = len(trndata_loader)
    start_training_time = time.time()
    best_acc = 0.0
    for epoch in range(nepochs):
        model.train()
        for iteration, (imgs, visits, labels) in enumerate(trndata_loader):
            imgs = imgs.to(device)
            visits = visits.to(device)
            # idx_labels = labels.clone()
            labels = torch.from_numpy(np.array(labels)).long().to(device)

            output = model(imgs, visits)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration % 50 == 0:
                logger.info(
                ", ".join(
                        [
                            "Epoch: [{epoch}/{num_epochs}]",
                            "Step: [{iter}/{total_step}",
                            "Loss: {loss:.4f}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        epoch = epoch + 1, num_epochs = nepochs,
                        iter = iteration + 1, total_step = total_step,
                        loss = loss.item(),
                        lr = optimizer.param_groups[0]["lr"],
                        memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        time_spent = time.time() - start_training_time
        logger.info("Epoch:[{}/{}], Time spent {}, Time per epoch {:.4f} s".format(
            epoch + 1, nepochs, str(datetime.timedelta(seconds=time_spent)), time_spent / (epoch + 1)))

        # val
        model.eval()
        with torch.no_grad():
            val_loss = 0
            total = 0
            correct = 0
            for _, (imgs, visits, labels) in enumerate(valdata_loader):
                imgs = imgs.to(device)
                visits = visits.to(device)
                # idx_labels = labels.clone()
                labels = torch.from_numpy(np.array(labels)).long().to(device)
                
                output = model(imgs, visits)
                val_loss += criterion(output, labels)
                correct += accuracy_score(labels.cpu().data.numpy(),np.argmax(output.cpu().data.numpy(), axis=1),normalize=False)
                total += labels.size(0) 
        acc = 100 * correct / total
        logger.info("Epoch:[{}/{}], validation loss: {}, Validation acc@1: {}%".format(
            epoch + 1, nepochs, val_loss, acc))   

        scheduler.step(val_loss)
        if (epoch + 1) % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(epoch + 1))
        if acc > best_acc:
            best_acc = acc
            checkpointer.save("best_model")
    checkpointer.save("model_final")
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (nepochs)
        )
    )