from allennlp.common.util import pad_sequence_to_length
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import masked_mean, masked_softmax
import copy

from transformers import BertModel

from allennlp.modules import ConditionalRandomField

import torch
import math
import torch.nn.functional as F
import random


class LinearSoftmaxOutputLayer(torch.nn.Module):
    """Output layer consisting of a linear layer and softmax."""

    def __init__(self, in_dim, num_labels):
        super(LinearSoftmaxOutputLayer, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(in_dim, self.num_labels)

    def forward(self, x, mask, labels=None):
        """
        x: shape (batch, max_sequence, in_dim)
        mask: shape (batch, max_sequence)
        labels: shape (batch, max_sequence)
        """
        logits = self.classifier(x)  # Linear transformation
        probabilities = F.softmax(logits, dim=-1)  # Softmax pour normalisation

        outputs = {"logits": probabilities}

        if labels is not None:
            # Compute CrossEntropy Loss
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            outputs["loss"] = loss

        else:
            predicted_labels = torch.argmax(probabilities, dim=-1)  # Prendre la classe max
            outputs["predicted_label"] = predicted_labels

        return outputs


class LinearSoftmaxPerTaskOutputLayer(torch.nn.Module):
    """Output layer per task for multi-task learning."""

    def __init__(self, in_dim, tasks):
        super(LinearSoftmaxPerTaskOutputLayer, self).__init__()
        self.per_task_output = torch.nn.ModuleDict()

        for task in tasks:
            self.per_task_output[task.task_name] = LinearSoftmaxOutputLayer(in_dim=in_dim, num_labels=len(task.labels))

    def forward(self, task, x, mask, labels=None, output_all_tasks=False):


        output = self.per_task_output[task](x, mask, labels)

        if output_all_tasks:
            output["task_outputs"] = []
            assert labels is None
            for t, task_output in self.per_task_output.items():
                task_result = task_output(x, mask)
                task_result["task"] = t
                output["task_outputs"].append(task_result)

        return output


class CRFOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, num_labels):
        super(CRFOutputLayer, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(in_dim, self.num_labels)
        self.crf = ConditionalRandomField(self.num_labels)

    def forward(self, x, mask, labels=None):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        batch_size, max_sequence, in_dim = x.shape

        logits = self.classifier(x)
        # outputs = {}
        probabilities = F.softmax(logits, dim=-1)
        outputs = {"logits": probabilities}  # Include logits in the output dictionary

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask)
            loss = -log_likelihood
            outputs["loss"] = loss
        else:
            best_paths = self.crf.viterbi_tags(logits, mask)
            predicted_label = [x for x, y in best_paths]
            predicted_label = [pad_sequence_to_length(x, desired_length=max_sequence) for x in predicted_label]
            predicted_label = torch.tensor(predicted_label)
            outputs["predicted_label"] = predicted_label

            #log_denominator = self.crf._input_likelihood(logits, mask)
            #log_numerator = self.crf._joint_likelihood(logits, predicted_label, mask)
            #log_likelihood = log_numerator - log_denominator
            #outputs["log_likelihood"] = log_likelihood

        return outputs

class CRFPerTaskOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, tasks):
        super(CRFPerTaskOutputLayer, self).__init__()

        self.per_task_output = torch.nn.ModuleDict()
        for task in tasks:
            self.per_task_output[task.task_name] = CRFOutputLayer(in_dim=in_dim, num_labels=len(task.labels))


    def forward(self, task, x, mask, labels=None, output_all_tasks=False):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        output = self.per_task_output[task](x, mask, labels)
        if output_all_tasks:
            output["task_outputs"] = []
            assert labels is None
            for t, task_output in self.per_task_output.items():
                task_result = task_output(x, mask)
                task_result["task"] = t
                output["task_outputs"].append(task_result)
        return output

    def to_device(self, device1, device2):
        self.task_to_device = dict()
        for index, task in enumerate(self.per_task_output.keys()):
            if index % 2 == 0:
                self.task_to_device[task] = device1
                self.per_task_output[task].to(device1)
            else:
                self.task_to_device[task] = device2
                self.per_task_output[task].to(device2)

    def get_device(self, task):
        return self.task_to_device[task]



class AttentionPooling(torch.nn.Module):
    def __init__(self, in_features, dimension_context_vector_u=200, number_context_vectors=5):
        super(AttentionPooling, self).__init__()
        self.dimension_context_vector_u = dimension_context_vector_u
        self.number_context_vectors = number_context_vectors
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=self.dimension_context_vector_u, bias=True)
        self.linear2 = torch.nn.Linear(in_features=self.dimension_context_vector_u,
                                       out_features=self.number_context_vectors, bias=False)

        self.output_dim = self.number_context_vectors * in_features

    def forward(self, tokens, mask):
        #shape tokens: (batch_size, tokens, in_features)

        # compute the weights
        # shape tokens: (batch_size, tokens, dimension_context_vector_u)
        a = self.linear1(tokens)
        a = torch.tanh(a)
        # shape (batch_size, tokens, number_context_vectors)
        a = self.linear2(a)
        # shape (batch_size, number_context_vectors, tokens)
        a = a.transpose(1, 2)
        a = masked_softmax(a, mask)

        # calculate weighted sum
        s = torch.bmm(a, tokens)
        s = s.view(tokens.shape[0], -1)
        return s



class BertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(BertTokenEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model"])
        # state_dict_1 = self.bert.state_dict()
        # state_dict_2 = torch.load('/home/astha_agarwal/model/pytorch_model.bin')
        # for name2 in state_dict_2.keys():
        #    for name1 in state_dict_1.keys():
        #        temp_name = copy.deepcopy(name2)
        #       if temp_name.replace("bert.", '') == name1:
        #            state_dict_1[name1] = state_dict_2[name2]

        #self.bert.load_state_dict(state_dict_1,strict=False)

        self.bert_trainable = config["bert_trainable"]
        self.bert_hidden_size = self.bert.config.hidden_size
        self.cacheable_tasks = config["cacheable_tasks"]
        for param in self.bert.parameters():
            param.requires_grad = self.bert_trainable

    def forward(self, batch):
        documents, sentences, tokens = batch["input_ids"].shape

        if "bert_embeddings" in batch:
            return batch["bert_embeddings"]

        attention_mask = batch["attention_mask"].view(-1, tokens)
        input_ids = batch["input_ids"].view(-1, tokens)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # shape (documents*sentences, tokens, 768)
        bert_embeddings = outputs[0]

        #### break the large judgements into sentences chunk of given size. Do this while inference
        # chunk_size = 1024
        # input_ids = batch["input_ids"].view(-1, tokens)
        # chunk_cnt = int(math.ceil(input_ids.shape[0]/chunk_size))
        # input_ids_chunk_list = torch.chunk(input_ids,chunk_cnt)
        #
        # attention_mask_chunk_list = torch.chunk(attention_mask,chunk_cnt)
        # outputs = []
        # for input_ids,attention_mask in zip(input_ids_chunk_list,attention_mask_chunk_list):
        #     with torch.no_grad():
        #         output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #         output = output[0]
        #         #output = output[0].to('cpu')
        #     outputs.append(copy.deepcopy(output))
        #     torch.cuda.empty_cache()
        #
        # bert_embeddings = torch.cat(tuple(outputs))  #.to('cuda')

        if not self.bert_trainable and batch["task"] in self.cacheable_tasks:
            # cache the embeddings of BERT if it is not fine-tuned
            # to save GPU memory put the values on CPU
            batch["bert_embeddings"] = bert_embeddings.to("cpu")

        return bert_embeddings









########## VERSION 2 ###############

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from transformers import BertModel
from context_fusion import ConcatProjection, GatedAdd, FiLMModulation, CrossAttentionFusion, ConditionalLayerNorm, \
    ReZeroFusion, AttentiveProtoFusion, AttentiveProtoSelector
import torch, torch.nn as nn, json, os
from collections import defaultdict


class BertHSLN(nn.Module):
    """
    Model for Baseline, Progressive Context, and Attention-based Context.
    """
    def __init__(self, config, tasks):
        super(BertHSLN, self).__init__()

        # Nouveaux arguments d'ablation
        self.use_word_lstm = config.get("use_word_lstm", True)
        self.use_attention_pooling = config.get("use_attention_pooling", True)

        self.use_all_prototypes = config.get("use_all_prototypes", True)
        # self.max_proto_per_sent = config.get("max_proto_per_sent", 13)
        self.max_proto_per_sent = len(tasks[0].labels) - 1
        # print(f"Nombre de prototypes: {self.max_proto_per_sent}")

        # 🔹 Configuration du modèle
        self.bert = BertTokenEmbedder(config)
        self.dropout = nn.Dropout(config["dropout"])
        self.lstm_hidden_size = config["word_lstm_hs"]
        self.strategy = config["strategy"]
        self.window_size = config["window_size"]
        self.style = config["sentence_attention_style"]
        self.window_size_att = config.get("window_size_att", 1)
        self.num_random = config.get("num_random", 2)
        self.global_k = config.get("global_k", 2)

        self.use_crf = config.get("use_crf", True)
        self.use_sentence_lstm = config.get("use_sentence_lstm", True)
        self.generic_output_layer = config.get("generic_output_layer")
        self.representation_dump_path = config.get("save_representations_to", "reps_output.jsonl")



        # Word-level LSTM (qu’on activera seulement si use_word_lstm=True)
        if self.use_word_lstm:
            self.word_lstm = PytorchSeq2SeqWrapper(nn.LSTM(
                input_size=self.bert.bert_hidden_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=1, batch_first=True, bidirectional=True
            ))
            word_lstm_out_dim = 2 * self.lstm_hidden_size
        else:
            # Pas de LSTM => la dimension reste celle de BERT
            word_lstm_out_dim = self.bert.bert_hidden_size

        # Attention Pooling (pour agrégation token->phrase)
        self.attention_pooling = AttentionPooling(
            in_features=word_lstm_out_dim,
            dimension_context_vector_u=config["att_pooling_dim_ctx"],
            number_context_vectors=config["att_pooling_num_ctx"]
        )

        embedding_dim_sentence = self.attention_pooling.output_dim # à vérifier

        # dimensions
        self.d_sent = self.attention_pooling.output_dim
        self.d_ctx = config.get("centroid_dim", self.d_sent)

        # injection
        self.ctx_fusion = config.get("ctx_fusion", "concat_proj")  # concat_proj | gated_add | none
        self.ctx_pos = config.get("ctx_position", "pre")

        # ─── après avoir défini self.d_sent et self.d_ctx ─────────────
        # quelle dimension en entrée du fusor ?
        if self.use_sentence_lstm:
            self.phrase_out_dim = 2 * self.lstm_hidden_size
        else:
            self.phrase_out_dim = self.d_sent  # pas de BiLSTM phrase

        # on choisit la dim selon ctx_position
        fusor_in_dim = self.d_sent if self.ctx_pos == "pre" else self.phrase_out_dim

        # 🔁 strat attentif si use_all_prototypes = True
        self.proto_selector = AttentiveProtoSelector(self.d_sent, self.d_ctx) if self.use_all_prototypes else None

        if self.ctx_fusion == "concat_proj":
            self.fusor = ConcatProjection(fusor_in_dim, self.d_ctx)
        elif self.ctx_fusion == "gated_add":
            self.fusor = GatedAdd(fusor_in_dim, self.d_ctx)
        elif self.ctx_fusion == "film":
            self.fusor = FiLMModulation(fusor_in_dim, self.d_ctx, config.get("film_hidden"))
        elif self.ctx_fusion == "cross_attn":
            self.fusor = CrossAttentionFusion(fusor_in_dim, self.d_ctx, config.get("cross_attn_mid", 1024),
                                              config.get("cross_attn_heads", 8))
        elif self.ctx_fusion == "cln":
            self.fusor = ConditionalLayerNorm(fusor_in_dim, self.d_ctx, config.get("cln_hidden"))
        else:
            self.fusor = None

        # ---- optionnel : ReZero sur le fusor choisi ---------------------------
        if self.fusor is not None and config.get("use_rezero", False):
            self.fusor = ReZeroFusion(self.fusor)
            print("i'm in zero ")
        else:
            self.fusor = self.fusor

        self.init_sentence_enriching(config)
        self.reinit_output_layer(tasks)

        # ------------ Charge banque de centroïdes
        self.doc2centroid = defaultdict(dict)  # doc_id → sent_idx → tensor

        centroid_paths = {"train" : f"{config['centroid_paths']}/train.jsonl",
                          "dev" : f"{config['centroid_paths']}/dev.jsonl",
                          "test" : f"{config['centroid_paths']}/test.jsonl"}

        for split, path in centroid_paths.items():
            if not os.path.exists(path): continue

            with open(path) as f:
                for line in f:
                    o = json.loads(line)


                    if self.use_all_prototypes:
                        self.doc2centroid[o["doc_id"]][o["sentence_idx"]] = [
                            torch.tensor(p["vector"], dtype=torch.float32) for p in
                            o["prototypes"][:self.max_proto_per_sent]
                        ]

                    else:
                        self.doc2centroid[o["doc_id"]][o["sentence_idx"]] = torch.tensor(o["centroid_vector"],
                                                                                         dtype=torch.float32)
        # -----------------------------------------------------------------

    def _build_all_proto_tensor(self, doc_names, S, device):
        B = len(doc_names)
        out = torch.zeros(B, S, self.max_proto_per_sent, self.d_ctx, device=device)
        for b, doc_id in enumerate(doc_names):
            sent_protos = self.doc2centroid.get(doc_id, {})
            for s_idx, proto_list in sent_protos.items():
                if s_idx >= S or not isinstance(proto_list, list): continue
                for p_idx, vec in enumerate(proto_list):
                    if p_idx < self.max_proto_per_sent:
                        out[b, s_idx, p_idx] = vec.to(device)
        return out

    def _build_centroid_tensor(self, doc_names, S, device):
        """Retourne (B,S,d_ctx)"""
        B = len(doc_names)
        out = torch.zeros(B, S, self.d_ctx, device=device)
        for b, doc_id in enumerate(doc_names):
            v = self.doc2centroid.get(doc_id, {})
            for s_idx, vec in v.items():
                if s_idx < S:
                    out[b, s_idx] = vec.to(device)
        return out

    def init_sentence_enriching(self, config):
        """
        Initialise le LSTM de phrase si `use_sentence_lstm` est activé.
        Initialise aussi les composants des stratégies de contexte.
        """
        input_dim = self.attention_pooling.output_dim

        if self.use_attention_pooling == False:
            input_dim = 2 * self.lstm_hidden_size

        if self.strategy == "baseline" and self.use_sentence_lstm:
                self.sentence_lstm = PytorchSeq2SeqWrapper(nn.LSTM(
                    input_size=input_dim,
                    hidden_size=self.lstm_hidden_size,
                    num_layers=1, batch_first=True, bidirectional=True
                ))

    def reinit_output_layer(self, tasks):
        """
        Initialise la couche de sortie (CRF ou Linear-Softmax) en fonction des paramètres.
        Adapte dynamiquement `input_dim` selon `use_sentence_lstm`.
        """


        input_dim = self.lstm_hidden_size * 2 if self.use_sentence_lstm else self.attention_pooling.output_dim
        if self.strategy == "self_attention_context":
            input_dim = self.attention_pooling.output_dim

        if self.use_crf:
            if self.generic_output_layer:
                self.output_layer = CRFOutputLayer(in_dim=input_dim, num_labels=len(tasks[0].labels))
            else:
                self.output_layer = CRFPerTaskOutputLayer(input_dim, tasks)
        else:
            if self.generic_output_layer:
                self.output_layer = LinearSoftmaxOutputLayer(in_dim=input_dim, num_labels=len(tasks[0].labels))
            else:
                self.output_layer = LinearSoftmaxPerTaskOutputLayer(input_dim, tasks)

    def forward(self, batch, labels=None, output_all_tasks=False):
        """
        Passe avant du modèle avec support pour différentes stratégies
        de contexte + injection dynamique des vecteurs centroïdes.
        """
        documents, sentences, tokens = batch["input_ids"].shape

        # ─── 1. Encodage token->phrase (inchangé) ─────────────────────────
        bert_embeddings = self.dropout(self.bert(batch))

        if self.use_word_lstm:
            tok_mask = batch["attention_mask"].view(-1, tokens)
            bert_embeddings = self.word_lstm(bert_embeddings, tok_mask)

        if self.use_attention_pooling:
            tok_mask = batch["attention_mask"].view(-1, tokens)
            sentence_embeddings = self.attention_pooling(bert_embeddings, tok_mask)
        else:
            sentence_embeddings = bert_embeddings[:, -1, :]

        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        sentence_embeddings = self.dropout(sentence_embeddings)

        # ─── 2. Préparation / création du tenseur centroid_vec ────────────
        #  a) déjà présent dans le batch  ➜ on le prend
        #  b) absent mais on a doc_name   ➜ on le fabrique à la volée
        ctx_vec = batch.get("centroid_vec")  # (B,S,d_ctx) ou None

        if ctx_vec is None and "doc_name" in batch:
            if self.use_all_prototypes:
                print(f"[INFO] Using all prototypes")
                all_proto_tensor = self._build_all_proto_tensor(batch["doc_name"], sentences,
                                                                sentence_embeddings.device)
                ctx_vec = self.proto_selector(sentence_embeddings, all_proto_tensor)
            else:
                ctx_vec = self._build_centroid_tensor(batch["doc_name"], sentences, sentence_embeddings.device)

        # si malgré tout ctx_vec == None  ➜ on reste en mode baseline
        if ctx_vec is not None:
            ctx_vec = ctx_vec.to(sentence_embeddings.device)

        # option PRE-fusion
        if ctx_vec is not None and self.ctx_pos == "pre":
            print("I'm in PRE ADDING CONTEXT ===========")
            sentence_embeddings = self.fusor(sentence_embeddings, ctx_vec)

        # ─── 3. LSTM de phrases / baseline ────────────────────────────────
        sentence_mask = batch["sentence_mask"]
        print(f"Shape of sentence : {sentence_embeddings.shape}")
        sent_repr = self.apply_default_context(sentence_embeddings, sentence_mask)

        # Sauvegarde des représentations pendant l'inférence
        if labels is None and "doc_name" in batch and self.representation_dump_path:
            try:
                with open(self.representation_dump_path, "a", encoding="utf-8") as f:
                    for b in range(sent_repr.size(0)):  # pour chaque document
                        doc_id = batch["doc_name"][b]
                        for s in range(sent_repr.size(1)):  # pour chaque phrase
                            if sentence_mask[b, s] == 1:
                                record = {
                                    "doc_id": doc_id,
                                    "sentence_id": s,
                                    "embedding": sent_repr[b, s].detach().cpu().tolist()
                                }
                                f.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"[⚠️] Erreur lors de la sauvegarde des représentations : {e}")

        # option POST-fusion
        if ctx_vec is not None and self.ctx_pos == "post":
            print("I'm in POST ADDING CONTEXT ===========")

            sent_repr = self.fusor(sent_repr, ctx_vec)

        # ─── 4. Tête de sortie ────────────────────────────────────────────
        if self.generic_output_layer:
            return self.output_layer(sent_repr, sentence_mask, labels)

        return self.output_layer(
            batch["task"], sent_repr, sentence_mask, labels, output_all_tasks
        )

    def apply_default_context(self, sentence_embeddings, sentence_mask):
        """
        Applique un simple LSTM de phrase si `use_sentence_lstm` est activé.
        """
        if self.use_sentence_lstm:
            return self.dropout(self.sentence_lstm(sentence_embeddings, sentence_mask))
        return self.dropout(sentence_embeddings)  # Utilisation directe des embeddings





