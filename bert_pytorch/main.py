import argparse
import torch

from torch.utils.data import DataLoader

from model.bert import BERT
from trainer.pretrain import BERTTrainer, BERTTrainerDual
from dataset.dataset import BERTDataset, BERTDatasetDual

from vocab_builder import WordVocab

torch.manual_seed(1)

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-val", "--val_dataset", type=str, default=None, help="val set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence len")
    parser.add_argument("-d", "--dropout", type=float, default=0.0, help="dropout rate")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=2000, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=False, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=2.5e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.00, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--load_pretrain", type=int, default=0, help="load pretrain")
    parser.add_argument("--dual_mask", type=int, default=0, help="dual mask")
    parser.add_argument("--debug", type=int, default=0, help="debug")

    args = parser.parse_args()

    # Logging Parameter
    print(args)

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))
    if args.dual_mask == 1:
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = BERTDatasetDual(args.train_dataset, vocab, seq_len=args.seq_len,
                                    corpus_lines=args.corpus_lines, on_memory=args.on_memory)

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = BERTDatasetDual(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
            if args.test_dataset is not None else None

        print("Creating Dataloader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) \
            if test_dataset is not None else None

        if args.val_dataset is not None:
            val_dataset = BERTDatasetDual(args.val_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory)
            val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    else:
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                    corpus_lines=args.corpus_lines, on_memory=args.on_memory)

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
            if args.test_dataset is not None else None

        print("Creating Dataloader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) \
            if test_dataset is not None else None

        if args.val_dataset is not None:
            val_dataset = BERTDataset(args.val_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory)
            val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)


    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)

    # Loading pretrain model
    if args.load_pretrain==1:
        print("Loading Pretrain Model", args.output_path + ".best" + ".ep-1")
        bert = torch.load(args.output_path + ".best" + ".ep-1")
    
    print("Creating BERT Trainer")
    if args.dual_mask == 1:
        trainer = BERTTrainerDual(bert, len(vocab), train_dataloader=train_data_loader, val_dataloader=val_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)
    else:
        trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, val_dataloader=val_data_loader, test_dataloader=test_data_loader,
                            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    print("Training Start")
    best_acc = 0.0
    if args.debug==0:
        for epoch in range(args.epochs):
            trainer.train(epoch)
            if epoch % 100 == 0:
                trainer.save(epoch, args.output_path)

            if epoch % 10 == 0:
                if val_data_loader is not None:
                    val_acc = trainer.val(epoch)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        trainer.save(-1, args.output_path + ".best")
    else:
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # if epoch % 100 == 0:
            #     trainer.save(epoch, args.output_path)

            if epoch % 10 == 0:
                if val_data_loader is not None:
                    val_acc = trainer.val(epoch)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        # trainer.save(-1, args.output_path + ".debug")

    print("Testing start")
    # test the best epoch
    bert = torch.load(args.output_path + ".best" + ".ep-1")
    tester = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, val_dataloader=train_data_loader, test_dataloader=test_data_loader,
                         lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                         with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)
    if test_data_loader is not None:
        tester.test(0)
    print("All Done!")

# Call train() function
if __name__ == "__main__":
    train()