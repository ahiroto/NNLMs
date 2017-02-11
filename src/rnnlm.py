#!/usr/bin/env python
"""
Script of Recurrent Neural Network language model.
"""

import argparse
from collections import OrderedDict
import logging
import sys
import time
import numpy as np
import six

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from RNN_block import RNN

# Set data
vocab = {}
Dict = {}


def set_vocab(filename, vocab_size):

    # word count
    sum_word = 0
    words = open(filename).read().replace('\n', ' </s> ').strip().split()
    for word in words:
        if word not in Dict:
            Dict[word] = 1
        else:
            Dict[word] += 1
        sum_word += 1
    print('{} words were loaded'.format(sum_word))
    print('Vocab size of data = {}'.format(len(Dict)))

    DictRaking = OrderedDict(sorted(Dict.items(),
                                    key=lambda x: x[1],
                                    reverse=True))
    WordCount = DictRaking.keys()

    # generate vocabulary
    if vocab_size == 0:
        vocab_size = len(Dict)
    for i in six.moves.range(vocab_size):
        vocab[WordCount[i]] = len(vocab)
    vocab['<unk>'] = -1
    print('Vocabulary generated. vocab_size = {}'.format(len(vocab)))


def load_data(filename):

    n_eos = 0
    n_unk = 0
    words = open(filename).read().replace('\n', ' </s> ').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word == '</s>':
            n_eos += 1
        if word not in vocab:
            dataset[i] = vocab['<unk>']
            n_unk += 1
        else:
            dataset[i] = vocab[word]
    return dataset, n_unk, n_eos


class RNNLM(chainer.Chain):

    def __init__(self, n_vocab, n_units, train=True, dropout=False):
        super(RNNLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units, ignore_label=-1),
            l1=RNN(n_units, n_units),
            l2=L.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train
        self.dropout = dropout

    def reset_state(self):
        self.l1.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        if self.dropout:
            h1 = self.l1(F.dropout(h0, train=self.train))
            y = self.l2(F.dropout(h1, train=self.train))
        else:
            h1 = self.l1(h0)
            y = self.l2(h1)
        return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t',
                        default='sample/train_set.txt',
                        help='Filename for the train set')
    parser.add_argument('--valid', '-d',
                        default='sample/dev_set.txt',
                        help='Filename for the validation set')
    parser.add_argument('--eval', '-e',
                        default='sample/eval_set.txt',
                        help='Filename for the test set')
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=8,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--debug', action='store_true',
                        help='Keep a debug log information')
    parser.set_defaults(debug=False)
    parser.add_argument('--dropout', action='store_true',
                        help='Use dropout')
    parser.set_defaults(dropout=False)
    parser.add_argument('--evaluation', action='store_true',
                        help='Run only evaluation (using test set)')
    parser.set_defaults(evaluation=False)
    parser.add_argument('--epoch', '-E', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--l2coeff', '-n', type=float, default=1e-5,
                        help='Coefficient of l2-regularization(WeightDecay)')
    parser.add_argument('--lr', '-L', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--out', '-o', default='.',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initial model for training')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of RNN units in each layer')
    parser.add_argument('--vocabsize', '-v', type=int, default=0,
                        help='Vocabulary size (default=fullsize vocabulary)')
    args = parser.parse_args()

    np.random.seed(10000)
    # logger setup
    if args.evaluation:
        logging.basicConfig(filename='%s/hidden%d_bptt%d_lr%f_batch%d'
                                     '_epoch%d_rnnlm_eval.log'
                                     % (args.out,
                                        args.unit,
                                        args.bproplen,
                                        args.lr,
                                        args.batchsize,
                                        args.epoch))
    else:
        logging.basicConfig(filename='%s/hidden%d_bptt%d_lr%f_batch%d'
                                     '_epoch%d_rnnlm.log'
                                     % (args.out,
                                        args.unit,
                                        args.bproplen,
                                        args.lr,
                                        args.batchsize,
                                        args.epoch))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    if args.debug:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    # Check params
    logger.debug('-----------------------------')
    logger.debug('hidden unit = %d \nBPTT = %d \nleaning rate = %.1f'
                 % (args.unit, args.bproplen, args.lr))
    logger.debug('-----------------------------')

    # Vocabulary setup
    set_vocab(args.train, args.vocabsize)
    eos_id = vocab['</s>'] if not args.trainwoeos else -2

    # Load data as np.ndarray(dtype=np.int32)
    train_data, train_unk, train_eos = \
        load_data(args.train, args.trainwoeos)
    valid_data, valid_unk, valid_eos = \
        load_data(args.valid, args.trainwoeos)
    test_data, test_unk, test_eos = \
        load_data(args.eval, args.trainwoeos)

    n_vocab = len(vocab)
    logger.info('# of vocab = %d' % n_vocab)

    if args.test:
        train_data = train_data[:100]
        valid_data = valid_data[:50]
        test_data = test_data[:50]

    logger.info('train_data size = %d' % train_data.size)
    logger.info('valid_data size = %d' % valid_data.size)
    logger.info('test_data  size = %d' % test_data.size)

    logger.debug('train unk words = %d' % train_unk)
    logger.debug('valid unk words = %d' % valid_unk)
    logger.debug('test  unk words = %d' % test_unk)

    logger.debug('train eos count = %4d, %.2f %%'
                 % (train_eos, float(train_eos)/train_data.size*100))
    logger.debug('valid eos count = %4d, %.2f %%'
                 % (valid_eos, float(valid_eos)/valid_data.size*100))
    logger.debug('test  eos count = %4d, %.2f %%'
                 % (test_eos, float(test_eos)/test_data.size*100))

    # Prepare an RNNLM model
    rnn = RNNLM(n_vocab, args.unit, dropout=args.dropout)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # Calculate perplexity only
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    xp = model.xp
    cuda.cupy.random.seed(10000)

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=args.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.l2coeff))

    # Init/Resume
    if args.initmodel:
        logger.info('Load model from %s' % args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.resume:
        logger.info('Load optimizer state from %s' % args.resume)
        chainer.serializers.load_npz(args.resume, optimizer)

    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    eval_rnn.train = False

    if not args.evaluation:
        # Trainning loop
        whole_len = train_data.shape[0]  # Size of train data
        jump = whole_len // args.batchsize    # Jump range
        cur_log_perp = xp.zeros(())
        epoch = 0
        start_at = time.time()
        cur_at = start_at
        accum_loss = 0
        batch_idxs = list(range(args.batchsize))
        logger.info('going to train %d iterations' % (jump * args.epoch))

        valid_prev = 10000
        for i in six.moves.range(jump * args.epoch):
            x = chainer.Variable(xp.asarray(
                [train_data[(jump * j + i) % whole_len]
                 for j in batch_idxs]))
            t = chainer.Variable(xp.asarray(
                [train_data[(jump * j + i + 1) % whole_len]
                 for j in batch_idxs]))
            loss_i = model(x, t)  # Compute loss
            accum_loss += loss_i  # Compute accum_loss
            cur_log_perp += loss_i.data

            if (i + 1) % args.bproplen == 0:  # Run truncated BPTT
                model.zerograds()
                accum_loss.backward()
                accum_loss.unchain_backward()  # Truncate
                accum_loss = 0
                optimizer.update()

            if (i + 1) % 10000 == 0:
                now = time.time()
                throuput = 10000. / (now - cur_at)
                perp = xp.exp(float(cur_log_perp) / 10000)
                logger.info('iter %d training perplexity: %.2f '
                            '(%.2f iters/sec)' % (i + 1, perp, throuput))
                cur_at = now
                cur_log_perp.fill(0)

            if (i + 1) % jump == 0:
                epoch += 1
                logger.info('evaluate')
                now = time.time()
                eval_rnn.reset_state()
                eval_rnn.sum_eos_proba = 0
                sum_log_perp = 0
                eos_log_perp = 0
                eos_count = 0
                unk_count = 0
                for i in six.moves.range(valid_data.size - 1):
                    x = chainer.Variable(xp.asarray(valid_data[i:i + 1]))
                    t = chainer.Variable(xp.asarray(valid_data[i + 1:i + 2]))
                    if valid_data[i + 1] == -1:
                        unk_count += 1
                    loss = eval_model(x, t)
                    if valid_data[i + 1] == eos_id:
                        eos_log_perp += loss.data
                        eos_count += 1
                    sum_log_perp += loss.data
                valid_log = \
                    float(sum_log_perp) / (valid_data.size - 1 - unk_count)
                valid_perp = xp.exp(valid_log)
                logger.info('epoch %d validation perplexity: %.2f'
                            % (epoch, valid_perp))
                if not args.trainwoeos:
                    eos_perp = \
                        xp.exp(float(eos_log_perp) / eos_count) \
                        if eos_count > 0 else 0
                    logger.info('epoch %d eos perplexity: %.2f'
                                % (epoch, eos_perp))
                # NewBob
                if valid_prev < valid_log * 1.003:
                    optimizer.lr *= 0.5
                    logger.info('Learning rate was decayed as %f'
                                % optimizer.lr)
                valid_prev = valid_log

                # logger.info('Save the current model')
                # chainer.serializers.save_npz(
                #     '{}/hidden{}_bptt{}_lr{}_batch{}_epoch{}_rnnlm.model.tmp'
                #     .format(args.out,
                #             args.unit,
                #             args.bproplen,
                #             args.lr,
                #             args.batchsize,
                #             epoch),
                #     model)
                # logger.info('Save the current optimizer')
                # chainer.serializers.save_npz(
                #     '{}/hidden{}_bptt{}_lr{}_batch{}_epoch{}_rnnlm.state.tmp'
                #     .format(args.out,
                #             args.unit,
                #             args.bproplen,
                #             args.lr,
                #             args.batchsize,
                #             epoch),
                #     optimizer)
                cur_at += time.time() - now  # skip time of evaluation

            sys.stdout.flush()
        # Save the model and the optimizer
        logger.info('Save the model')
        chainer.serializers.save_npz(
            '{}/rnnlm.model'
            .format(args.out),
            model)
        logger.info('Save the optimizer')
        chainer.serializers.save_npz(
            '{}/rnnlm.opt'
            .format(args.out),
            optimizer)

    # Evaluate on test dataset
    logger.info('Test')
    eval_rnn.reset_state()  # initialize state
    sum_log_perp = 0
    unk_count = 0
    for i in six.moves.range(test_data.size - 1):
        x = chainer.Variable(xp.asarray(test_data[i:i + 1]))
        t = chainer.Variable(xp.asarray(test_data[i + 1:i + 2]))
        loss = eval_model(x, t)
        if test_data[i + 1] == -1:
            unk_count += 1
        sum_log_perp += loss.data

    test_perp = \
        xp.exp(float(sum_log_perp) /
               (test_data.size - 1 - unk_count))
    logger.info('Test perplexity: %.2f' % test_perp)

if __name__ == '__main__':
    main()
