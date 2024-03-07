import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader

from model.language_model import BERTLM, BERT, BERTLM_Dual
from trainer.optimizer.optim_schedule import ScheduledOptim
from trainer.optimizer.adamw import AdamW

class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, val_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        print("USING OUR MODEL")
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.masked_criterion = nn.NLLLoss(ignore_index=0)
        self.next_criterion = nn.NLLLoss()

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        return self.iteration(epoch, self.train_data)

    def val(self, epoch):
        return self.iteration(epoch, self.val_data, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        if train:
            self.model.train()
            for i, data in enumerate(data_loader):
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward the next_sentence_prediction and masked_lm model
                next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

                # 2-1. NLL(negative log likelihood) loss of is_next classification result
                next_loss = self.next_criterion(next_sent_output, data["is_next"])

                # 2-2. NLLLoss of predicting masked token word
                mask_loss = self.masked_criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

                # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
                loss = next_loss + mask_loss

                # 3. backward and optimization only in train
                if train:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                # next sentence prediction accuracy
                correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
                avg_loss += loss.item()
                total_correct += correct
                total_element += data["is_next"].nelement()

                post_fix = {
                    "epoch": epoch,
                    "iter": "[%d/%d]" % (i, len(data_loader)),
                    "avg_loss": avg_loss / (i + 1),
                    "mask_loss": mask_loss.item(),
                    "next_loss": next_loss.item(),
                    "avg_next_acc": total_correct / total_element * 100,
                    "loss": loss.item()
                }

                if i % self.log_freq == 0:
                    print(post_fix)

                    # Logging for PaperSpace matrix monitor
                    # index = epoch * len(data_loader) + i
                    # for code in ["avg_loss", "mask_loss", "next_loss", "avg_next_acc"]:
                    #     print(json.dumps({"chart": code, "y": post_fix[code], "x": index}))

            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_loader), "total_acc=",
                total_correct * 100.0 / total_element)
            return total_correct * 100.0 / total_element
        else:
            self.model.eval()
            with torch.inference_mode():
                for i, data in enumerate(data_loader):
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    data = {key: value.to(self.device) for key, value in data.items()}

                    # 1. forward the next_sentence_prediction and masked_lm model
                    next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

                    # 2-1. NLL(negative log likelihood) loss of is_next classification result
                    next_loss = self.next_criterion(next_sent_output, data["is_next"])

                    # 2-2. NLLLoss of predicting masked token word
                    mask_loss = self.masked_criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

                    # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
                    loss = next_loss + mask_loss

                    # next sentence prediction accuracy
                    correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
                    avg_loss += loss.item()
                    total_correct += correct
                    total_element += data["is_next"].nelement()

                    post_fix = {
                        "epoch": epoch,
                        "iter": "[%d/%d]" % (i, len(data_loader)),
                        "avg_loss": avg_loss / (i + 1),
                        "mask_loss": mask_loss.item(),
                        "next_loss": next_loss.item(),
                        "avg_next_acc": total_correct / total_element * 100,
                        "loss": loss.item()
                    }

                    if i % self.log_freq == 0:
                        print(post_fix)

                        # Logging for PaperSpace matrix monitor
                        # index = epoch * len(data_loader) + i
                        # for code in ["avg_loss", "mask_loss", "next_loss", "avg_next_acc"]:
                        #     print(json.dumps({"chart": code, "y": post_fix[code], "x": index}))

                print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_loader), "total_acc=",
                    total_correct * 100.0 / total_element)
                return total_correct * 100.0 / total_element
    
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

class BERTTrainerDual:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, val_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM_Dual(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        print("USING OUR MODEL")
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.masked_criterion = nn.NLLLoss(ignore_index=0)
        self.next_criterion = nn.NLLLoss()

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        return self.iteration(epoch, self.train_data)

    def val(self, epoch):
        return self.iteration(epoch, self.val_data, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        avg_loss = 0.0
        avg_loss_mask = 0.0
        avg_loss_next = 0.0
        avg_loss_kd = 0.0
        total_correct = 0
        total_element = 0
        if train:
            self.model.train()

            for i, data in enumerate(data_loader):
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward the next_sentence_prediction and masked_lm model
                next_sent_output, mask_lm_output, next_sent_output2, mask_lm_output2, loss_kd = self.model.forward(data["bert_input"], data["segment_label"], data["bert_input2"], data["segment_label2"])

                # 2-1. NLL(negative log likelihood) loss of is_next classification result
                next_loss = self.next_criterion(next_sent_output, data["is_next"])
                next_loss2 = self.next_criterion(next_sent_output2, data["is_next2"])

                # 2-2. NLLLoss of predicting masked token word
                mask_loss = self.masked_criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
                mask_loss2 = self.masked_criterion(mask_lm_output2.transpose(1, 2), data["bert_label2"])

                # print("loss_kd", loss_kd)
                # print("SHAPE KD", loss_kd.shape)
                # print("NEXT loss shape", next_loss.shape)
                # print("MASK loss shape", mask_loss.shape)
                # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
                loss_kd = loss_kd.mean()
                loss = next_loss + mask_loss + next_loss2 + mask_loss2 + 100*loss_kd.mean()

                # 3. backward and optimization only in train
                if train:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                # next sentence prediction accuracy
                correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
                avg_loss += loss.item()
                avg_loss_mask += mask_loss.item()
                avg_loss_next += next_loss.item()
                avg_loss_kd += loss_kd.item() 
                total_correct += correct
                total_element += data["is_next"].nelement()

                post_fix = {
                    "epoch": epoch,
                    "iter": "[%d/%d]" % (i, len(data_loader)),
                    "avg_loss": avg_loss / (i + 1),
                    "avg_loss_kd": avg_loss_kd / (i + 1),
                    "avg_loss_mask": avg_loss_mask / (i + 1),
                    "avg_loss_next": avg_loss_next / (i + 1),
                    "mask_loss": mask_loss.item(),
                    "next_loss": next_loss.item(),
                    "avg_next_acc": total_correct / total_element * 100,
                    "loss": loss.item()
                }

                if i % self.log_freq == 0:
                    print(post_fix)

                    # Logging for PaperSpace matrix monitor
                    # index = epoch * len(data_loader) + i
                    # for code in ["avg_loss", "mask_loss", "next_loss", "avg_next_acc"]:
                    #     print(json.dumps({"chart": code, "y": post_fix[code], "x": index}))

            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_loader), "total_acc=",
                total_correct * 100.0 / total_element)
            
            return total_correct * 100.0 / total_element
        else:
            self.model.eval()
            # torch inference mode
            with torch.inference_mode():
                for i, data in enumerate(data_loader):
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    data = {key: value.to(self.device) for key, value in data.items()}

                    # 1. forward the next_sentence_prediction and masked_lm model
                    next_sent_output, mask_lm_output, next_sent_output2, mask_lm_output2, loss_kd = self.model.forward(data["bert_input"], data["segment_label"], data["bert_input2"], data["segment_label2"])

                    # 2-1. NLL(negative log likelihood) loss of is_next classification result
                    next_loss = self.next_criterion(next_sent_output, data["is_next"])
                    next_loss2 = self.next_criterion(next_sent_output2, data["is_next2"])

                    # 2-2. NLLLoss of predicting masked token word
                    mask_loss = self.masked_criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
                    mask_loss2 = self.masked_criterion(mask_lm_output2.transpose(1, 2), data["bert_label2"])

                    # print("loss_kd", loss_kd)
                    # print("SHAPE KD", loss_kd.shape)
                    # print("NEXT loss shape", next_loss.shape)
                    # print("MASK loss shape", mask_loss.shape)
                    # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
                    loss = next_loss + mask_loss + next_loss2 + mask_loss2 + 100*loss_kd.mean()

                    # next sentence prediction accuracy
                    correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
                    avg_loss += loss.item()
                    avg_loss_mask += mask_loss.item()
                    avg_loss_next += next_loss.item()
                    avg_loss_kd += loss_kd.item() 
                    total_correct += correct
                    total_element += data["is_next"].nelement()

                    post_fix = {
                        "epoch": epoch,
                        "iter": "[%d/%d]" % (i, len(data_loader)),
                        "avg_loss": avg_loss / (i + 1),
                        "avg_loss_kd": avg_loss_kd / (i + 1),
                        "avg_loss_mask": avg_loss_mask / (i + 1),
                        "avg_loss_next": avg_loss_next / (i + 1),
                        "mask_loss": mask_loss.item(),
                        "next_loss": next_loss.item(),
                        "avg_next_acc": total_correct / total_element * 100,
                        "loss": loss.item()
                    }

                    if i % self.log_freq == 0:
                        print(post_fix)

                        # Logging for PaperSpace matrix monitor
                        # index = epoch * len(data_loader) + i
                        # for code in ["avg_loss", "mask_loss", "next_loss", "avg_next_acc"]:
                        #     print(json.dumps({"chart": code, "y": post_fix[code], "x": index}))

                print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_loader), "total_acc=",
                    total_correct * 100.0 / total_element)
                
                return total_correct * 100.0 / total_element
    
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path