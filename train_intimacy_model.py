# train.py

from config import Config
import math
from scipy import stats
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch import nn, Tensor
from torch.autograd import Variable
import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from argparse import ArgumentParser


torch.manual_seed(0)




def padding(text, pad, max_len = 50):

    return text if len(text) >= max_len else (text + [pad] * (max_len-len(text)))

def encode_batch(text, berts, max_len = 50):
    tokenizer = berts[0]
    t1 = []
    for line in text:
        t1.append(padding(tokenizer.encode(line,add_special_tokens = True, max_length = max_len,truncation=True),tokenizer.pad_token_id,max_len))
    return t1


def data_iterator(train_x, train_y, batch_size = 64):
    n_batches = math.ceil(len(train_x) / batch_size)
    for idx in range(n_batches):
        x = train_x[idx *batch_size:(idx+1) * batch_size]
        y = train_y[idx *batch_size:(idx+1) * batch_size]
        yield x, y
        

def get_metrics(model, test_x, test_y, config, tokenizer, test = False, save_path='test_prediction_final.txt'):
    cuda = config.cuda
    all_preds = []
    test_iterator = data_iterator(test_x, test_y, batch_size=64)
    all_y = []
    all_x = []
    model.eval()
    for x, y in test_iterator:
        ids = encode_batch(x, (tokenizer,model), max_len = config.max_len)
        with torch.no_grad():
            if cuda:
                input_ids = Tensor(ids).cuda().long()                
                labels = torch.cuda.FloatTensor(y)
            else:
                input_ids = Tensor(ids).long()
                labels = torch.FloatTensor(y)
            outputs = model(input_ids, labels=labels)
            loss, y_pred = outputs[:2]

        predicted = y_pred.cpu().data
        all_preds.extend(predicted.numpy())
        all_y.extend(y)
        all_x.extend(x)

    all_res = np.array(all_preds).flatten()
    if test and save_path:
        with open(save_path, 'w') as w:
            for i in range(len(all_y)):
                if i < 2:
                    print(all_x[i], all_res[i], test_y[i])
                w.writelines(all_x[i] + '\t' + str(all_y[i]) + '\t' + str(all_res[i]) + '\n')

    #print('pearson r:', stats.pearsonr(all_res, all_y)[0])
    score = 0
    return loss,stats.pearsonr(all_res, all_y)[0]

def run_epoch(model, train_data, val_data, tokenizer,config, optimizer):
    train_x, train_y = train_data[0], train_data[1]
    val_x, val_y = val_data[0], val_data[1]
    iterator = data_iterator(train_x, train_y, config.batch_size)
    train_losses = []
    val_accuracies = []
    losses = []

    for i, (x,y) in tqdm(enumerate(iterator),total=int(len(train_x)/config.batch_size)):
        #print('iteration', i)
        model.zero_grad()

        ids = encode_batch(x, (tokenizer,model), max_len = config.max_len)


        if config.cuda:
            input_ids = Tensor(ids).cuda().long()
            labels = torch.cuda.FloatTensor(y)
        else:
            input_ids = Tensor(ids).long()
            labels = torch.FloatTensor(y)

        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        loss.backward()
        #print('train_loss',loss)
        losses.append(loss.data.cpu().numpy())
        optimizer.step()

        if (i + 1) % 1 == 0:
            #print("Iter: {}".format(i))
            avg_train_loss = np.mean(losses)
            train_losses.append(avg_train_loss)
            #print("\tAverage training loss: {:.5f}".format(avg_train_loss))
            losses = []

            # Evalute Accuracy on validation set
            model.eval()
            all_preds = []
            val_iterator = data_iterator(val_x, val_y, config.batch_size)
            for x, y in val_iterator:
                ids = encode_batch(x, (tokenizer,model), max_len = config.max_len)
                #x = Variable(Tensor(x))

                with torch.no_grad():

                    if config.cuda:
                        input_ids = Tensor(ids).cuda().long()
                        labels = torch.cuda.FloatTensor(y)
                    else:
                        input_ids = Tensor(ids).long()
                        labels = torch.FloatTensor(y)
                    outputs = model(input_ids, labels=labels)
                    loss, y_pred = outputs[:2]

                predicted = y_pred.cpu().data

                all_preds.extend(predicted.numpy())

            
            all_res = np.array(all_preds).flatten()            
            score = (np.square(val_y - all_res)).mean()
            val_accuracies.append(score)
            model.train()

    return train_losses, val_accuracies

def get_test_result(model, test_x, test_y, config,tokenizer, save_path, ext_test = False, pure_inference=False):
    cuda = config.cuda
    all_raw = []
    all_preds = []
    all_y = []
    all_x = []
    test_iterator = data_iterator(test_x, test_y, batch_size=256)
    model.eval()
    i = 0
    for x, y in test_iterator:
        print(str(i * 256) + '/' + str(len(test_x)))
        i += 1
        #print(x[:5])
        ids = encode_batch(x, (tokenizer, model), max_len = config.max_len)
        # x = Variable(Tensor(x))

        with torch.no_grad():
            if cuda:
                input_ids = Tensor(ids).cuda().long()
                #labels = torch.cuda.FloatTensor(y)
            else:
                input_ids = Tensor(ids).long()
                #labels = torch.FloatTensor(y)
            outputs = model(input_ids)
            y_pred = outputs[0]

        predicted = y_pred.cpu().data
        #predicted = torch.max(y_pred.cpu().data, 1)[1]
        all_preds.extend(predicted.numpy())
        #all_raw.extend(y_pred.cpu().data.numpy())
        all_y.extend(y)
        all_x.extend(x)


    #all_res = [1 if i[0] > 0 else -1 for i in all_preds]
    all_res = np.array(all_preds).flatten()
    #all_raw = np.array(all_raw)

    if save_path:
        with open(save_path, 'w') as w:
            if pure_inference:
                 for i in range(len(all_y)):                
                    if i < 2:
                        print(all_x[i], all_res[i])                    
                    w.writelines(all_x[i] + '\t' + str(all_res[i]) + '\n')
            else:
                for i in range(len(all_y)):                
                    if i < 2:
                        print(all_x[i], all_res[i], test_y[i])                    
                    w.writelines(all_x[i] + '\t' + str(all_y[i]) + '\t' + str(all_res[i]) + '\n')
    
    if not pure_inference:
        print('mse:', (np.square(all_y - all_res)).mean())
        print('pearson r:', stats.pearsonr(all_res, all_y)[0])

    if ext_test:
        print('book pearson r:', stats.pearsonr(all_res[:50], all_y[:50])[0])
        print('twitter pearson r:', stats.pearsonr(all_res[50:100], all_y[50:100])[0])
        print('movie pearson r:', stats.pearsonr(all_res[100:150], all_y[100:150])[0])

    return all_res, all_y




def arguments():
    parser = ArgumentParser()
    parser.set_defaults(show_path=False, show_similarity=False)

    parser.add_argument('--mode')
    parser.add_argument('--model_name')
    parser.add_argument('--pre_trained_model_name_or_path')
    parser.add_argument('--train_path', default='train.txt')
    parser.add_argument('--val_path', default='val.txt')
    parser.add_argument('--test_path', default='test.txt')
    parser.add_argument('--predict_data_path')
    parser.add_argument('--model_saving_path', default=None)
    parser.add_argument('--test_saving_path', default=None)

    return parser.parse_args()


if __name__=='__main__':

    args = arguments()
    def get_data(path):
        print('open:',path)
        text = []
        y = []
        with open(path, 'r') as r:
            lines = r.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                text.append(line[0])
                if len(line) == 2:
                    y.append(float(line[1]))
                elif len(line) == 1:
                    y.append(0.0)
                else:
                    print('error in loading:',path)
        return text,y

    
    config = Config(args.model_name)
    tokenizer = RobertaTokenizer.from_pretrained(args.pre_trained_model_name_or_path,num_labels=1,output_attentions = False,output_hidden_states = False)
    # Create Model with specified optimizer and loss function
    ##############################################################

    model = RobertaForSequenceClassification.from_pretrained(args.pre_trained_model_name_or_path,num_labels=1,output_attentions = False,output_hidden_states = False)
    if config.cuda:
        model.cuda()
        
       

    if args.mode == 'train':
        
        train_text,  train_labels = get_data(args.train_path)
        val_text,  val_labels = get_data(args.val_path)
        test_text,  test_labels = get_data(args.test_path)
        
        train_x = train_text
        train_y = np.array(train_labels)
        val_x = val_text
        val_y = np.array(val_labels)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-6)
        ##############################################################

        train_data = [train_x, train_y]
        val_data = [val_x, val_y]

        # Get Accuracy of final model
        test_x = test_text
        test_y = np.array(test_labels)
        best_val = 100.0
        best_test = 100.0
        best_r = 100

        for i in range(config.max_epochs):
            print ("Epoch: {}".format(i))

            train_losses,val_accuracies = run_epoch(model, train_data, val_data, tokenizer,config, optimizer)
            test_acc,test_r = get_metrics(model, test_x, test_y, config, tokenizer, test = True, save_path=args.test_saving_path)
            #print('Final Test Accuracy: {:.4f}'.format(test_acc))

            print("\tAverage training loss: {:.5f}".format(np.mean(train_losses)))
            print("\tAverage Val MSE: {:.4f}".format(np.mean(val_accuracies)))
            if np.mean(val_accuracies) < best_val:
                best_val = np.mean(val_accuracies)
                best_test = test_acc
                best_r = test_r
                if i >= 1 and args.model_saving_path:
                    model.save_pretrained(args.model_saving_path)
                    tokenizer.save_pretrained(args.model_saving_path)


        print('model saved at', args.model_saving_path)
        print('best_val_loss:', best_val)
        print('best_test_loss:',best_test)
        print('best_test_pearsonr:',best_r)

        #test_acc = get_metrics(model, test_x, test_y, config, tokenizer, test = True)
        #if args.model_saving_path:
        #    model.save_pretrained(args.model_saving_path)
        #    tokenizer.save_pretrained(args.model_saving_path)


    elif args.mode == 'internal-test':
        val_text,  val_labels = get_data(args.val_path)
        test_text,  test_labels = get_data(args.test_path)
        final_test_text,final_test_y = get_data(args.predict_data_path)
        
        print('external_test:')
        test_result, test_score = get_test_result(model, final_test_text, final_test_y, config, tokenizer,save_path=args.test_saving_path, ext_test = True)

        print('val:')
        test_result, test_score = get_test_result(model, val_text, val_labels, config, tokenizer,save_path=None, ext_test = False)

        print('test:')
        test_result, test_score = get_test_result(model, test_text, test_labels, config, tokenizer,save_path=None, ext_test = False)
        
    elif args.mode == 'inference':
        final_test_text,final_test_y = get_data(args.predict_data_path)
        test_result, test_score = get_test_result(model, final_test_text, final_test_y, config, tokenizer,save_path=args.test_saving_path, ext_test = False, pure_inference=True)





