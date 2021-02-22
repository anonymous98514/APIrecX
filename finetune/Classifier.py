from dataloader.DataloaderApi import *
import torch.nn as nn
import torch.optim
import time
import difflib
import torch.nn.functional as F
import datetime
from collections import Counter
from GPT.RAdam.radam.radam import RAdam
from math import log
from numpy import array
from numpy import argmax
from collections import defaultdict
import copy
import re

class Candidate:
    def __init__(self, pre_ids, pro, is_complete):
        self.pre_ids = pre_ids
        self.pro = pro
        self.is_complete = is_complete


class BestToken:
    def __init__(self, pre_ids, pro):
        self.pre_ids = pre_ids
        self.pro = pro


class Classifier:
    def __init__(self, model, model_lstm, args, vocab, word_vocab, Rvocab, tokenizer):
        self.model = model
        self.model_lstm = model_lstm
        self.vocab = vocab
        self.args = args
        self.word_vocab = word_vocab
        self.Rvocab = Rvocab
        self.counter = Counter()
        self.next_api = 0
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.ep = 0
        self.control_num = 0

    # 打印模型参数
    def summary(self):
        print(self.model)

    # 训练
    def train(self, train_data, dev_data, test_data, args_device, arg, epoch, search_word_dict, train_batch_len=None,
              dev_batch_len=None):
        # optimizer = torch.optim.Adam(self.model.parameters(),
        #                              lr=self.args.lr,
        #                              weight_decay=self.args.weight_decay)
        # optimizer = RAdam(self.model.parameters(),  lr=self.args.lr,
        #                              weight_decay=self.args.weight_decay)
        optimizer = RAdam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                          weight_decay=self.args.weight_decay)
        # criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id).to(self.device)

        train_loss_list, train_acc_list = [], []
        best_acc = 0
        dev_loss_list, dev_acc_list = [], []
        patenice = 0
        for ep in range(self.args.epoch):
            self.ep = ep
            train_data_num = 1
            train_num = 1
            train_acc = 0
            train_acc_1 = 0
            train_loss = 0
            word_acc = 0
            # word_num = 0
            start_time = datetime.datetime.now()
            self.model.train()
            print("start train")
            # print(len(train_data))
            for onebatch in get_batch_train(train_data, self.args.batch_size, arg, train_batch_len):
                words, tags, mask, seq_lengths = batch_numberize(onebatch, args_device, arg)

                targets = words[:, 1:].contiguous()

                pred = self.model(words)
                pred = pred[:, :-1].contiguous()

                loss = self.compuate_loss(targets.view(-1), pred.view(-1, pred.shape[-1]))
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                train_loss += loss.data.item()
                acc_1, num = self.compuate_acc(targets.view(-1), pred.view(-1, pred.shape[-1]))
                train_acc += acc_1
                train_data_num += num

            end_time = datetime.datetime.now()
            during_time = end_time - start_time
            print (len(dev_data))
            dev_acc, dev_loss, dev_data_num, dev_word_acc, dev_num, perplexity = self.validate(dev_data, args_device,
                                                                                               arg, dev_batch_len,
                                                                                               False)

            train_acc /= train_data_num
            train_loss /= train_data_num

            # train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            # dev_acc_list.append(dev_acc)
            if patenice > 2:
                self.ep = self.ep - patenice

                break
            if dev_acc > best_acc:
                print(dev_acc, best_acc)
                dev_acc_list.append(dev_acc)
                best_acc = dev_acc_list[-1]
                patenice = 0
            else:
                patenice += 1
                print(patenice)


            print("[Epoch {}] train loss :{} train_acc:{} %  Time:{}  train_word_acc:{} train data num:{} train data vocab num:{}".format(
                ep + 1, train_loss, train_acc * 100, during_time, word_acc / train_num, train_data_num, train_num))
            print(
                "[Epoch {}] dev loss :{} dev_acc:{} dev_word_acc:{} % train data num:{} train data vocab num:{} perplexity:{}".format(ep + 1,
                                                                                                          dev_loss,
                                                                                                          dev_acc * 100,
                                                                                                          dev_word_acc / dev_num,
                                                                                                          dev_data_num,
                                                                                                          dev_num,
                                                                                                          perplexity))


        s = datetime.datetime.now()
        print(datetime.datetime.now() - s)
        # print(test_num)

        if arg.is_save:
        #     torch.save(self.model.state_dict(), "data/API/trained_GPT_{}_jdbc_bpe".format(-1))
            torch.save(self.model, "data/API/data/trained_GPT_{}_swing_bpe_1".format(-1))
        # print("{} round:training complete".format(epoch))
        return dev_acc_list[-1], perplexity, self.ep

    def validate(self, dev_data, args_device, arg, batch_len, is_refine):

        dev_loss = 0
        dev_acc = 0
        batch_num = 1
        dev_data_num = 1
        perplexity = 0.0
        dev_word_acc = 0
        num_1 = 1
        num_2 = 0
        self.model.eval()
        dev_num = 1
        refine_num = 0
        no_refine_num = 0
        true_test_seq = []
        pred_api_seq = []
        acc_2 = 0
        acc_3 = 0
        # with torch.no_grad:
        for line_num, onebatch in enumerate(get_batch_train(dev_data, arg.batch_size, arg, None)):
            batch_num += 1
            # refine_pred = []
            # refine_pred  = torch.FloatTensor(1,self.args.max_seq_len,4565).zero_().to(args_device)
            words, tags, mask, seq_lengths = batch_numberize(onebatch, args_device, arg)
            # print(onebatch[0])
            # pred:[batch_size, seq_len, hidden_size * num_direction]
            # pred = self.model(words)
            targets = words[:, 1:].contiguous()
            # print(words)
            # print(targets)
            pred = self.model(words)
            pred = pred[:, :-1].contiguous()
            loss = self.compuate_loss(targets.view(-1), pred.view(-1, pred.shape[-1]))
            dev_loss += loss.data.item()
            acc_1, num = self.compuate_acc(targets.view(-1), pred.view(-1, pred.shape[-1]))
            perplexity += torch.exp(loss).data.item()
            dev_acc += acc_1
            dev_data_num += num

        print(acc_2, acc_3, dev_word_acc)
        dev_acc /= dev_data_num
        dev_loss /= dev_data_num
        perplexity /= batch_num
        print("num_1:", num_1)
        print("num_2:", num_2)
        return dev_acc, dev_loss, dev_data_num, dev_word_acc, num_1, perplexity

    # 评估模型
    def evluate(self, dev_data,dev_data_1, args_device, arg, batch_len, is_refine, search_word_dict):
        reject_token = ["[EOS]","[BOS]","[PAD]","[UNK]"]
        appendControlNodesStrings = [
            "IF", "CONDITION", "THEN", "ELSE",
            "WHILE", "BODY",
            "TRY", "TRYBLOCK", "CATCH", "FINALLY",
            "FOR", "INITIALIZATION", "COMPARE", "UPDATE",
            "FOREACH", "VARIABLE", "ITERABLE",
        ]
        length_pro = 0
        length_pro_1 = 0
        length_pro_2 = 0
        control_node = 0
        k = 10
        top1_len = Counter()
        top1_ground_true_len = Counter()
        top1_len_info = Counter()
        top3_len = Counter()
        top3_ground_true_len = Counter()
        beam_size = arg.boundary
        batch_num = 0
        dev_data_num = 0
        perplexity = 0.0
        dev_word_acc_top1 = 0
        dev_word_acc_top10 = 0
        dev_word_acc_top3 = 0
        dev_word_acc_top5 = 0
        dev_word_acc_class_1 = 0
        dev_word_acc_class_3 = 0
        dev_word_acc_class_5 = 0
        dev_word_acc_class_10 = 0
        #non-control-node-num
        num_1 = 1
        #control-node-num
        num_2 = 0
        #coreect_non-control-node-num
        num_3 = 0
        num_4 = 0
        num_5 = 0
        new_num = 0
        self.model.eval()
        dev_num = 1
        refine_num = 0
        no_refine_num = 0
        domain_count = 1
        a_api_counter = set()
        b_api_counter = set()
        c_api_counter = set()
        cross_domain = 0
        for line_num, onebatch in enumerate(get_batch_train(dev_data_1, 1, arg, None)):

            # print("*********new_seq***********")
            words, tags, mask, seq_lengths = batch_numberize(onebatch, args_device, arg)
            targets = words[:, 1:].contiguous()
            true_seq = "".join(self.tokenizer.convert_ids_to_tokens(onebatch[0].input_ids))

            pred_index_1 = 0
            pred_index = 0
            for word_loc, word_len in enumerate(onebatch[0].word_index):
                cur_word = words.contiguous().clone()
                cur_word = cur_word.expand(beam_size, 512)
                true_token = []
                if word_loc == 0:
                    true_token1 = [self.tokenizer.convert_id_to_token(targets[0, index:index + 1].item()) for index
                                   in range(word_len)]
                    pred_index_1 = word_len
                    pred_index = word_len
                    # print("number one word", "".join(true_token1).replace("▁", ""))
                    continue

                pred_index = pred_index_1
                varible_cut_dot = 0
                for word_dex in range(word_len):
                    if self.tokenizer.convert_id_to_token(targets[0, pred_index_1].item()) == "▁.":
                        varible_cut_dot = word_dex + 1
                    true_token.append(
                        self.tokenizer.convert_id_to_token(targets[0, pred_index_1:pred_index_1 + 1].item()))
                    pred_index_1 += 1
                true_api = "".join(true_token).replace("▁", "")
                if onebatch[0].tags[word_loc] != 1:
                    continue
                else:
                    if true_api.find(".new") != -1:
                        continue
                    else:
                        b_api_counter.add(true_api)

        for line_num, onebatch in enumerate(get_batch_train(dev_data, 1, arg, None)):
            batch_num += 1
            words, tags, mask, seq_lengths = batch_numberize(onebatch, args_device, arg)
            targets = words[:, 1:].contiguous()
            if is_refine:
                true_seq = "".join(self.tokenizer.convert_ids_to_tokens(onebatch[0].input_ids))
                cahe_list =true_seq.replace("[BOS]","").replace("[PAD]","").replace("▁","").replace("</t>"," ").replace("[EOS]","").split(" ")
                pred_index_1 = 0
                pred_index = 0
                for word_loc, word_len in enumerate(onebatch[0].word_index):

                    candidate_list = []
                    bestToken_list = []
                    beam_candidate_list = []
                    tokensDone = 0
                    iter = 0
                    count = 0
                    token_pro_sum = 0.0
                    # if The probability of the best candidate is less than the worst current complete top-k tokens
                    hope = True

                    cur_word = words.contiguous().clone()
                    cur_word = cur_word.expand(beam_size, 512)
                    true_token = []
                    if word_loc == 0:
                        true_token1 = [self.tokenizer.convert_id_to_token(targets[0, index:index + 1].item()) for index
                                       in range(word_len)]
                        pred_index_1 = word_len
                        pred_index = word_len
                        # print("number one word", "".join(true_token1).replace("▁", ""))
                        continue

                    if word_loc <= 3:
                        cur_cahe_list = cahe_list[:word_loc]
                    else:
                        cur_cahe_list = cahe_list[word_loc-3:word_loc]
                    class_cahe_list = [class_method.split(".")for class_method in cur_cahe_list]
                    # dev_num += 1
                    pred_index = pred_index_1
                    varible_cut_dot = 0
                    for word_dex in range(word_len):
                        if self.tokenizer.convert_id_to_token(targets[0, pred_index_1].item()) =="▁.":
                            varible_cut_dot = word_dex+1
                        true_token.append(
                            self.tokenizer.convert_id_to_token(targets[0, pred_index_1:pred_index_1 + 1].item()))
                        pred_index_1 += 1
                    true_api = "".join(true_token).replace("▁", "")

                    if true_api.find(".new") != -1:
                        if onebatch[0].tags[word_loc] == 1:
                            new_num += 1
                        continue
                    dev_num += 1
                    if onebatch[0].tags[word_loc] == 1:
                        domain_count += 1
                        c_api_counter.add(true_api)
                    else:
                        continue
                    # else:
                    #     continue
                    a = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>", "", true_api)
                    true_api_nop = re.sub(u"\\(\\)|\\{|\\[\\]|\\>|\\<", "", a)
                    method_len = word_len - varible_cut_dot
                    class_name = [words[0, index].item() for index in range(pred_index+1,pred_index+varible_cut_dot+1)]
                    if len(class_name) == 0:
                        class_name_var ="".join(self.tokenizer.convert_ids_to_tokens(class_name)).replace("▁", "")
                    else:
                        class_name_var = "".join(self.tokenizer.convert_ids_to_tokens(class_name)).replace("▁", "")
                    pred_index = pred_index+varible_cut_dot

                    append_info = [[words[0, pred_index].item(), 1]]


                    while ((tokensDone <= 5000) and hope):
                        iter += 1

                        if len(beam_candidate_list) > 1:
                            if count >= k:
                                break

                            for i in range(len(beam_candidate_list)):
                                if pred_index+ len(beam_candidate_list[i].pre_ids)>= 512:
                                    print("over the limit")
                                    count += 1
                                    continue
                                cur_word[i, pred_index:pred_index + len(beam_candidate_list[i].pre_ids)] = torch.tensor(
                                    beam_candidate_list[i].pre_ids, dtype=torch.long)
                                currt_pred = self.model(cur_word[i:i + 1, :])
                                singel_word_pred = currt_pred[:, pred_index + len(beam_candidate_list[i].pre_ids)-1, :].clone()
                                singel_word_pred = F.softmax(singel_word_pred, dim=1)
                                subword_pro_order = torch.argsort(singel_word_pred, dim=1, descending=True)[0][:beam_size]

                                for pred_subword in subword_pro_order:

                                    if self.tokenizer.convert_id_to_token(pred_subword.item()).find("</t>") != -1 or self.tokenizer.convert_id_to_token(pred_subword.item()) in reject_token:
                                        # print('-------------')
                                        if self.tokenizer.convert_id_to_token(pred_subword.item()) in reject_token:
                                            continue
                                        tokensDone += 1

                                        update_list = [index for index in beam_candidate_list[i].pre_ids]
                                        # print(candidate_list[i].pre_ids)
                                        update_list.append(pred_subword.item())
                                        method_name ="".join([self.tokenizer.convert_id_to_token(index) for index in
                                                 update_list]).replace("▁", "").replace("</t>", "").replace(
                                            "[EOS]", "").replace("[UNK]", "").replace("[PAD]", "").replace("[BOS]","")
                                        # if len(class_name) == 0:
                                        #     print(method_name)
                                        # print(class_name_var)
                                        if class_name_var != "":
                                            if method_name not in search_word_dict[class_name_var.replace(".","")]:
                                                continue
                                            bestToken_list.append(BestToken(update_list,
                                                                            beam_candidate_list[i].pro * singel_word_pred[0][
                                                                                pred_subword.item()].item()))
                                        else:
                                            if method_name not in appendControlNodesStrings:
                                                continue
                                            bestToken_list.append(BestToken(update_list,
                                                                            beam_candidate_list[i].pro *
                                                                            singel_word_pred[0][
                                                                                pred_subword.item()].item()))

                                        bestToken_list = sorted(bestToken_list, key=lambda x: x.pro, reverse=True)

                                        if len(bestToken_list) > k :
                                            bestToken_list.pop(-1)

                                    else:

                                        update_list = [index for index in beam_candidate_list[i].pre_ids]
                                        update_list.append(pred_subword.item())
                                        candidate_list.append(Candidate(update_list,
                                                                        beam_candidate_list[i].pro * singel_word_pred[0][
                                                                            pred_subword.item()].item(), False))

                            candidate_list = sorted(candidate_list, key=lambda x: x.pro, reverse=True)

                            token_pro_sum = sum([token.pro for token in bestToken_list])
                            if len(bestToken_list) >= 1 and len(candidate_list) != 0:
                                if candidate_list[0].pro < bestToken_list[-1].pro:
                                    hope = False

                            if len(candidate_list) < beam_size:
                                print (len(candidate_list))
                                for i in range(len(candidate_list), 0, -1):

                                    beam_candidate_list[i - 1] = candidate_list.pop(i - 1)
                            else:
                                for i in range(beam_size,0,-1):

                                    beam_candidate_list[i-1] = candidate_list.pop(i-1)

                        else:
                            cur_word[0, pred_index] = append_info[0][0]
                            currt_pred = self.model(cur_word[0:1, :])
                            init_candidate_list, init_bestTokens = self.compuate_acc_2(
                                currt_pred[0, pred_index:pred_index + 1, :], append_info, k, reject_token,search_word_dict,class_name_var,appendControlNodesStrings,beam_size=beam_size)
                            candidate_list = [data for data in init_candidate_list]
                            bestToken_list =[data for data in init_bestTokens]
                            beam_candidate_list = [data for data in init_candidate_list]

                            if len(bestToken_list) >= 1 and len(candidate_list) != 0:
                                if candidate_list[0].pro < bestToken_list[-1].pro:
                                    hope = False

                            pred_index += 1

                            for i in range(beam_size,0,-1):
                                candidate_list.pop(i-1)
                    final_result = []
                    final_result_check = []
                    final_result_nop = []
                    final_class_result = []

                    bestToken_list = sorted(bestToken_list, key=lambda x: x.pro, reverse=True)
                    for best_token in bestToken_list[:10]:

                        final_result.append(class_name_var+"".join([self.tokenizer.convert_id_to_token(index) for index in best_token.pre_ids]).replace("▁", "").replace("</t>","").replace("[EOS]","").replace("[UNK]","").replace("[PAD]",""))

                        final_result_check.append(best_token.pro)
                        final_class_result.append("".join(
                            [self.tokenizer.convert_id_to_token(index) for index in best_token.pre_ids]).replace("▁",
                                                                                                                 "").replace(
                            "</t>", "").replace("[EOS]", "").replace("[UNK]", "").split(".")[0])
                        raw_api = class_name_var + "".join([self.tokenizer.convert_id_to_token(index) for index in best_token.pre_ids]).replace("▁","").replace("</t>", "").replace("[EOS]", "").replace("[UNK]", "").replace("[PAD]", "")
                        a = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>", "", raw_api)
                        true_api_nop_can = re.sub(u"\\(\\)|\\{|\\[\\]|\\>|\\<", "", a)
                        final_result_nop.append(true_api_nop_can)
                    if true_api.replace("</t>","") in final_result[:1]:
                        dev_word_acc_top1 += 1
                        if onebatch[0].tags[word_loc] == 1:
                            dev_word_acc_class_1 += 1

                    else:
                        if true_api.replace("</t>","") not in appendControlNodesStrings:
                            pass

                        else:

                            control_node += 1
                            pass
                    if true_api.replace("</t>","") in final_result[:3]:
                        dev_word_acc_top3 += 1
                        if onebatch[0].tags[word_loc] == 1:
                            dev_word_acc_class_3 += 1
                    else:
                        m3 = 0
                    if true_api.replace("</t>", "") in final_result[:5]:
                        dev_word_acc_top5 += 1
                        if onebatch[0].tags[word_loc] == 1:
                            dev_word_acc_class_5 += 1

                    if true_api.replace("</t>", "") in final_result:
                        if true_api.replace("</t>", "") not in appendControlNodesStrings:
                            num_3 += 1
                        else:
                            num_4 += 1
                        dev_word_acc_top10 += 1
                        if true_api not in b_api_counter:
                            if onebatch[0].tags[word_loc] == 1:
                                a_api_counter.add(true_api)
                                cross_domain += 1
                        if onebatch[0].tags[word_loc] == 1:
                            dev_word_acc_class_10 += 1
                    else:
                        if true_api.replace("</t>", "") not in appendControlNodesStrings:
                            num_1 += 1
                            if true_api_nop.replace("</t>", "") in final_result_nop:
                                pass
                            else:
                                if onebatch[0].tags[word_loc] == 1:
                                    num_5 += 1


                        else:
                            num_2 += 1


        word_acc_1 = dev_word_acc_top1 / dev_num
        word_acc_3 = dev_word_acc_top3 / dev_num
        word_acc_5 = dev_word_acc_top5 / dev_num
        word_acc_10 = dev_word_acc_top10 / dev_num
        dev_1 = dev_word_acc_class_1 / domain_count
        dev_3 = dev_word_acc_class_3 / domain_count
        dev_5 = dev_word_acc_class_5 / domain_count
        dev_10 = dev_word_acc_class_10 / domain_count

        print("class acc: top1:{}   top3:{}    top5:{}    top10:{}".format(dev_1,dev_3,dev_5,dev_10))

        d_api_counter = b_api_counter & c_api_counter
        print("------------------------------")
        print (dev_num)
        print (domain_count)
        print("cross domain")
        print(a_api_counter)
        print (len(a_api_counter))
        print (cross_domain /domain_count )
        if len(c_api_counter) == 0:
            print ("coverage:",len(d_api_counter)/ 1)
        else:
            print("coverage:", len(d_api_counter) / len(c_api_counter))

        return word_acc_1,word_acc_3,word_acc_5,word_acc_10, dev_num,[dev_1,dev_3,dev_5,dev_10],domain_count






    def compuate_acc(self, true_tags, logit):

        correct_num = 0


        select_index = []
        for i in range(logit.shape[0]):
            if true_tags[i].item() != 0:
                # prediction[i] = logit[i]
                select_index.append(i)
        if len(select_index) == 0:
            # print(true_tags)
            return 0, 0
        # print(len(select_index))

        logit = torch.index_select(logit, 0, torch.tensor(select_index).long().to(self.args.device))
        true_tags = torch.index_select(true_tags, 0, torch.tensor(select_index).long().to(self.args.device))
        logit = F.softmax(logit, dim=1)
        for i in range(logit.shape[0]):
            if true_tags[i] in torch.argsort(logit[i], descending=True)[: 2]:
                correct_num += 1


        return correct_num, true_tags.shape[0]

    def compuate_acc_1(self, true_tags, logit):

        correct_num = 0
        select_index = []
        append_info = []
        for i in range(true_tags.shape[0]):
            select_index.append(i)


        logit = torch.index_select(logit, 0, torch.tensor(select_index).long().to(self.args.device))
        true_tags = torch.index_select(true_tags, 0, torch.tensor(select_index).long().to(self.args.device))
        logit = F.softmax(logit, dim=1)

        for i in range(logit.shape[0]):
            if true_tags[i].item() in torch.argsort(logit[i], descending=True)[: 5].tolist():
                correct_num += 1


                for i in range(self.args.boundary):
                    append_info.append(torch.argsort(logit[-1], descending=True)[i].item())
            else:
                for i in range(self.args.boundary):
                    append_info.append(torch.argsort(logit[-1], descending=True)[i].item())

        return correct_num, true_tags.shape[0], append_info

    def compuate_acc_2(self, logit, pre_info, k, reject_token,search_dict,class_name,control_lable,beam_size,target=None):
        bestTokens = []
        pre_candidate = []
        lowest_pro = 0.0
        logit = F.softmax(logit, dim=1)
        sort = torch.argsort(logit, dim=1, descending=True)
        flag1 = False
        flag2 = False


        if len(pre_info) != 1:
            for i in range(logit.shape[0]):
                for j in range(self.args.boundary):
                    append_info.append((sort[-1][j].item() % self.tokenizer.vocab_size,
                                        logit[-1][sort[-1][j].item()].item() *
                                        pre_info[int(sort[-1][j].item() / self.tokenizer.vocab_size)][1]))
                    pre_candidate.append(pre_info[int(sort[-1][j].item() / self.tokenizer.vocab_size)][0])
        else:
            # for i in range(logit.shape[0]):
            for j in range(logit.shape[1]):
                if flag1 and flag2:
                    break

                if len(pre_candidate) < beam_size:
                    if self.tokenizer.convert_id_to_token(sort[0][j].item()).find(
                            "</t>") == -1 and self.tokenizer.convert_id_to_token(
                            sort[0][j].item()) not in reject_token:
                        pre_candidate.append(
                            Candidate([sort[0][j].item()], logit[0][sort[0][j].item()].item(), False))
                else:
                    flag1 = True
                if len(bestTokens) < k:
                    method_name = self.tokenizer.convert_id_to_token(sort[0][j].item()).replace("▁","")
                    if class_name == "":
                        # print(method_name)
                        if method_name.replace("</t>", "") in control_lable:
                            bestTokens.append(BestToken([sort[0][j].item()], logit[0][sort[0][j].item()].item()))

                    else:
                        if method_name.find(
                                "</t>") != -1 or method_name in reject_token:
                            if method_name in reject_token:
                                continue
                                # bestTokens.append(BestToken([sort[0][j].item()], logit[0][sort[0][j].item()].item()))

                            else:
                                if method_name.replace("</t>","") in search_dict[class_name]:
                                    bestTokens.append(BestToken([sort[0][j].item()], logit[0][sort[0][j].item()].item()))
                else:

                    flag2 = True


        bestTokens = sorted(bestTokens, key=lambda x: x.pro, reverse=True)

        return pre_candidate, bestTokens


    def compuate_loss(self, true_tags, logit):

        loss = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        loss = loss(logit, true_tags)

        return loss

    def is_validate(self, pred_sub_word_order, validate_class_name, search_word_dict):
        currt_api = validate_class_name[0]
        currt_api_index = 0
        currt_api_search_path = None
        is_complete = False
        flag = False

        for i in range(pred_sub_word_order.shape[0]):
            if flag:
                break

            for validate_class in validate_class_name[1]:
                if flag:
                    break
                for c_class in search_word_dict[validate_class]:
                    if c_class.find(
                            currt_api + self.tokenizer.convert_id_to_token(pred_sub_word_order[i].item()).replace("▁",
                                                                                                                  "")) != -1:

                        currt_api_index = pred_sub_word_order[i].item()
                        currt_api = currt_api + self.tokenizer.convert_id_to_token(pred_sub_word_order[i].item())
                        currt_api_search_path = validate_class

                        if self.tokenizer.convert_id_to_token(currt_api_index).find("▁") != -1:

                            currt_api = currt_api.replace("▁", "")
                            is_complete = True
                        flag = True
                        break
                    else:
                        continue

        return currt_api, is_complete, currt_api_index, currt_api_search_path

    def found_validate_class(self, pred_subword, all_classes, n):
        # validate_class_name = ()
        refine_words = []
        # print(all_classes)
        pred_subword = self.tokenizer.convert_id_to_token(pred_subword.item()).replace("▁", "")

        if len(pred_subword) < 4:
            refine_api = difflib.get_close_matches(pred_subword, all_classes.keys(), 100, 0.1)
            for can_dai in refine_api:
                if len(refine_words) == n:
                    break
                if can_dai.startswith(pred_subword, 0, len(pred_subword)) and len(can_dai) > 4:
                    refine_words.append(can_dai)
        else:
            refine_api = difflib.get_close_matches(pred_subword, all_classes.keys(), 100, 0.1)
            for can_dai in refine_api:
                if len(refine_words) == n:
                    break
                if can_dai.find(pred_subword) != -1:
                    refine_words.append(can_dai)
        if len(refine_api) == 0:
            refine_words = n * ["null"]
        else:
            for r_api in refine_api:
                refine_words.append(r_api)
            refine_words = refine_words + (n - len(refine_api)) * ["null"]
        return refine_words

    def refine(self, pred, true_tags):
        appendControlNodesStrings = [
            "IF", "CONDITION", "THEN", "ELSE",
            "WHILE", "BODY",
            "TRY", "TRYBLOCK", "CATCH", "FINALLY",
            "FOR", "INITIALIZATION", "COMPARE", "UPDATE",
            "FOREACH", "VARIABLE", "ITERABLE",
        ]
        tokens = []
        true_token = []
        flag = False
        refine_words = []
        pred = F.softmax(pred, dim=1)
        # print(true_tags.shape,pred.shape)

        for j in range(pred.shape[0]):
            true_token.append(self.tokenizer.convert_id_to_token(true_tags[j].item()).replace("▁", ""))
        for i in range(10):
            token = []
            for j in range(pred.shape[0]):
                top5 = torch.argsort(pred[j], descending=True)[: 10]
                token.append(self.tokenizer.convert_id_to_token(top5[i].item()).replace("▁", ""))

            word = "".join(token)
            tokens.append(word)

            # print(true_word)
            refine_word = difflib.get_close_matches(word, self.word_vocab, 1, 0.6)
            if len(refine_word) == 0:
                refine_words.append("null")
            else:
                refine_words.append(refine_word[0])
        true_word = "".join(true_token)
        if true_word in appendControlNodesStrings:
            self.control_num += 1
        # print(word, true_word, refine_word)
        # if len(refine_word) != 0:
        if true_word in refine_words:
            # print(true_word,tokens)
            # print(true_word, refine_words)
            flag = True
        else:
            # print(true_word, refine_words)
            pass

        return true_word, tokens


    def refine_for_rec(self, pred):
        appendControlNodesStrings = [
            "IF", "CONDITION", "THEN", "ELSE",
            "WHILE", "BODY",
            "TRY", "TRYBLOCK", "CATCH", "FINALLY",
            "FOR", "INITIALIZATION", "COMPARE", "UPDATE",
            "FOREACH", "VARIABLE", "ITERABLE",
        ]
        refine_words = []
        for word in pred:
            # print(word)
            refine_api = difflib.get_close_matches(word.replace("▁", ""), self.word_vocab, 10, 0.05)
            # try:
            if len(refine_api) == 0:
                refine_words.append("null")
            else:
                for r_api in refine_api:
                    refine_words.append(r_api)
            # except IndexError:

        return refine_words

    def beam_search_decoder(self, data, k):
        data = F.softmax(data, dim=1)
        # print(data.shape)
        sequences = [[list(), 1.0]]
        # walk over each step in sequence
        data = data.cpu().detach().numpy().tolist()
        for row in data:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    # print(row[j].item())
                    # a = -log(row[j].item())
                    try:
                        candidate = [seq + [j], score * -log(row[j])]
                        all_candidates.append(candidate)
                    except:
                        # print(row[j])
                        candidate = [seq + [j], score * -log(0.00001)]
                        all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            # select k best
            sequences = ordered[:k]
        return sequences
    def cache(self,top20_result,class_cahe_list):
        for i,best_token in enumerate(top20_result):
            token = "".join(self.tokenizer.convert_ids_to_tokens(best_token.pre_ids)).replace("▁", "").replace("</t>","").replace("[EOS]","").replace("[UNK]","").replace("[PAD]","")
            if token.split(".")[0] in class_cahe_list:
                # count = cur_cahe_list.count(token)
                top20_result[i].pro = 0.7 * top20_result[i].pro + 0.3

            else:
                top20_result[i].pro = 0.7 * top20_result[i].pro
                # count = 0

        return top20_result





