import time
import logging
import warnings
import torch
from tqdm import tqdm
from segmentation_models_pytorch.metrics import iou_score, get_stats

class Trainer:
    """Trainer

    Class that eases the training of a PyTorch model.

    Parameters
    ----------
    model : torch.Module
        The model to train.
    criterion : torch.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    device

    Attributes
    ----------
    train_loss_ : list
    val_loss_ : list
    train_step_loss : list
    val_step_loss : list
    """
    def __init__(self, model, criterion1,criterion2, testLoader, optimizer, device, modelPath, alpha=0.6, beta=0.4):
        self.model = model
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.alpha = alpha
        self.beta = beta
        self.optimizer = optimizer
        self.device = device
        self.modelpath = modelPath

        # send model to device
        self.model.to(self.device)
        
        self.test_X = 0
        self.test_Y = 0

        self.IOU_score = []

        # attributes
        self.train_loss_ = []
        self.val_loss_ = []

        self.train_step_loss = []
        self.val_step_loss = []

        self.test_X , self.test_Y = next(iter(testLoader))
        self.test_X , self.test_Y = self.test_X.to(self.device), self.test_Y.to(self.device)

        logging.basicConfig(level=logging.INFO)

    def fit(self, train_loader, val_loader, epochs, early_stopping=True, patience=5):
        """Fits.

        Fit the model using the given loaders for the given number
        of epochs.

        Parameters
        ----------
        train_loader :
        val_loader :
        epochs : int
            Number of training epochs.
        early_stopping : bool
            If True, enables early stopping.
        patience : int
            Number of epochs to wait for improvement before stopping.
        """
        # track total training time
        total_start_time = time.time()

        best_val_loss = float('inf')
        patience_counter = 0
        best_res_loss = 0

        print("Starting Training...")
        # ---- train process ----
        for epoch in range(epochs):
            # track epoch time
            epoch_start_time = time.time()

            # train
            tr_loss = self._train(train_loader,epoch)

            # validate
            #val_loss = self._validate(val_loader)
            val_loss = 0

            self.train_loss_.append(tr_loss)
            #self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time

            
            # IOU score
            IOUscore_ = self._test(self.model, self.test_X, self.test_Y)
            self.IOU_score.append(IOUscore_.item())


            # LOG
            self._logger(tr_loss, val_loss, IOUscore_.item(), epoch + 1, epochs, epoch_time)

            # save checkpoint
            filename = 'c_{}_IOU_{:.4f}.pth.tar'
            filename = self.modelpath + filename
            filename = filename.format(epoch, IOUscore_)
            torch.save(self.model.state_dict(), filename)

            # early stopping check
            if early_stopping:
                if IOUscore_.item() > best_res_loss:
                    best_res_loss = IOUscore_.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch + 1} due to lack of improvement.")
                    break
            
            

        total_time = time.time() - total_start_time

        # final message
        logging.info(f"End of training. Total time: {round(total_time, 5)} seconds")

        # Evalution graph



    def _train(self, loader, epoch):
        self.model.train()
        with tqdm(loader, unit="batch") as tepoch:
            for X, Y in tepoch:
                # Description
                tepoch.set_description(f"Epoch {epoch+1}")

                # move to device
                X, Y = self._to_device(X, Y, self.device)

                # forward pass
                out = self.model(X)

                # loss
                loss = self._compute_loss(out, Y)

                #Log
                self.train_step_loss.append(loss.cpu().detach().numpy())

                # remove gradient from previous passes
                self.optimizer.zero_grad()

                # backprop
                loss.backward()

                # parameters update
                self.optimizer.step()

                # Description
                tepoch.set_postfix(loss=loss.item())

        return loss.item()

    def _to_device(self, X, Y, device):
        return X.to(device), Y.to(device)

    def _validate(self, loader):
        self.model.eval()

        with torch.no_grad():
            for X, Y in loader:
                # move to device
                X, Y = self._to_device(X, Y, self.device)

                out = self.model(X)
                loss = self._compute_loss(out, Y)
                
                #Log
                self.val_step_loss.append(loss.cpu().detach().numpy())
        
        return loss.item()

    def _compute_loss(self, real, target):
        try:
            loss1 = self.criterion1(real, target)
            loss2 = self.criterion2(real, target)
            loss = loss1*self.alpha + loss2*self.beta
        except:
            loss = self.criterion(real, target.long())
            msg = f"Target tensor has been casted from"
            msg = f"{msg} {type(target)} to 'long' dtype to avoid errors."
            warnings.warn(msg)

        # DICE LOSS!
        
        return loss

    def _logger(self, tr_loss, val_loss, IOU, epoch, epochs, epoch_time):
        # to satisfy pep8 common limit of characters
        msg = f"Epoch {epoch}/{epochs} | Train loss: {tr_loss}"
        msg = f"{msg} | Validation loss: {val_loss}"
        msg = f"{msg} | IOU score: {IOU}"
        msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"

        logging.info(msg)

    def _test(self, model, X, mask):
        self.model.eval()
        output = torch.argmax(model(X), dim=1) 

        tp, fp, fn, tn = get_stats(output, mask, mode="multiclass",num_classes=10)
        iou_res= iou_score(tp, fp, fn, tn, reduction="micro")
        return iou_res
