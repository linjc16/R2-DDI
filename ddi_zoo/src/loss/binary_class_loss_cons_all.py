from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion
import torch
import math
import torch.nn.functional as F
from fairseq import metrics
from omegaconf import II
import numpy as np


@dataclass
class BinaryClassConsNegSigmoidPosV2Config(FairseqDataclass):
    classification_head_name: str = II("model.classification_head_name")
    consis_alpha: float = field(default=0.0)
    mt_alpha: float = field(default=1.0)
    p_consis_alpha: float = field(default=0.0)
    reg_loss_weight: float = field(default=0)

@register_criterion("binary_class_loss_cons_all", dataclass=BinaryClassConsNegSigmoidPosV2Config)
class BinaryClassNegConsSigmoidPosV2Criterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, consis_alpha, mt_alpha, p_consis_alpha, reg_loss_weight):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.consis_alpha = consis_alpha
        self.mt_alpha = mt_alpha
        self.p_consis_alpha = p_consis_alpha
        self.reg_loss_weight = reg_loss_weight

        print(f"trans_loss alpha: {reg_loss_weight}")

        acc_sum = torch.zeros(30)
        self.register_buffer('acc_sum', acc_sum)

    def build_input(self, sample, classification_head_name, neg=False):
        if not neg:
            return {
                'drug_a_seq': sample['drug_a_seq'] if 'drug_a_seq' in  sample else None,
                'drug_b_seq': sample['drug_b_seq'] if 'drug_b_seq' in  sample else None,
                'drug_a_graph': sample['drug_a_graph'] \
                    if "drug_a_graph" in sample else None,
                'drug_b_graph': sample['drug_b_graph'] \
                    if "drug_b_graph" in sample else None,
                'net_rel': sample['target'],
                'features_only': True,
                'classification_head_name': classification_head_name
                }
        else:
            return {
                'drug_a_seq': sample['drug_a_seq_neg'] if 'drug_a_seq_neg' in  sample else None,
                'drug_b_seq': sample['drug_b_seq_neg'] if 'drug_b_seq_neg' in  sample else None,
                'drug_a_graph': sample['drug_a_graph_neg'] \
                    if "drug_a_graph_neg" in sample else None,
                'drug_b_graph': sample['drug_b_graph_neg'] \
                    if "drug_b_graph_neg" in sample else None,
                'net_rel': sample['target'],
                'features_only': True,
                'classification_head_name': classification_head_name
            }

    def build_da_input(self, sample, classification_head_name, neg=False):
        if not neg:
            return {
                'drug_a_seq': sample['da_drug_a_seq'] if 'da_drug_a_seq' in  sample else None,
                'drug_b_seq': sample['da_drug_b_seq'] if 'da_drug_b_seq' in  sample else None,
                'drug_a_graph': sample['da_drug_a_graph'] \
                    if "da_drug_a_graph" in sample else None,
                'drug_b_graph': sample['db_drug_b_graph'] \
                    if "db_drug_b_graph" in sample else None,
                'net_rel': sample['target'],
                'features_only': True,
                'classification_head_name': classification_head_name
                }
        else:
            return {
                'drug_a_seq': sample['da_drug_a_seq_neg'] if 'da_drug_a_seq_neg' in  sample else None,
                'drug_b_seq': sample['da_drug_b_seq_neg'] if 'da_drug_b_seq_neg' in  sample else None,
                'drug_a_graph': sample['da_drug_a_graph_neg'] \
                    if "da_drug_a_graph_neg" in sample else None,
                'drug_b_graph': sample['da_drug_b_graph_neg'] \
                    if "drug_b_graph_neg" in sample else None,
                'net_rel': sample['target'],
                'features_only': True,
                'classification_head_name': classification_head_name
            }
    
    def build_db_input(self, sample, classification_head_name, neg=False):
        if not neg:
            return {
                'drug_a_seq': sample['db_drug_a_seq'] if 'db_drug_a_seq' in  sample else None,
                'drug_b_seq': sample['db_drug_b_seq'] if 'db_drug_b_seq' in  sample else None,
                'drug_a_graph': sample['db_drug_a_graph'] \
                    if "db_drug_a_graph" in sample else None,
                'drug_b_graph': sample['db_drug_b_graph'] \
                    if "db_drug_b_graph" in sample else None,
                'net_rel': sample['target'],
                'features_only': True,
                'classification_head_name': classification_head_name
                }
        else:
            return {
                'drug_a_seq': sample['db_drug_a_seq_neg'] if 'db_drug_a_seq_neg' in  sample else None,
                'drug_b_seq': sample['db_drug_b_seq_neg'] if 'db_drug_b_seq_neg' in  sample else None,
                'drug_a_graph': sample['db_drug_a_graph_neg'] \
                    if "db_drug_a_graph_neg" in sample else None,
                'drug_b_graph': sample['db_drug_b_graph_neg'] \
                    if "db_drug_b_graph_neg" in sample else None,
                'net_rel': sample['target'],
                'features_only': True,
                'classification_head_name': classification_head_name
            }
    

    def forward(self, model, sample, reduce=True):

        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        
        pos_input = self.build_input(sample, self.classification_head_name)
        pos_logits, pos_logits_deltaH, pos_logits_deltaT, pos_logits_ori = model.forward_cons_neg(**pos_input)
        neg_input = self.build_input(sample, self.classification_head_name, neg=True)
        neg_logits, neg_logits_deltaH, neg_logits_deltaT, neg_logits_ori = model.forward_cons_neg(**neg_input)


        targets = model.get_targets(sample['target'], None).view(-1)
        sample_size = targets.size(0)
        
        if isinstance(pos_logits, tuple):

            mix_logits, t_logits, closs = pos_logits
            mix_neg_logits, t_neg_logits, ncloss= neg_logits

            loss = (- F.logsigmoid(mix_logits).mean() - F.logsigmoid(-mix_neg_logits).mean() ) / 2.
            logging_out = {
                "loss": loss.data,
                "ntokens": sample["ntokens"] * 2,
                "nsentences": sample_size * 2,
                "sample_size": sample_size * 2,
                }
            
            t_loss = (- F.logsigmoid(t_logits).mean() - F.logsigmoid(-t_neg_logits).mean() ) / 2.
            loss += self.mt_alpha * t_loss
            
            inter_loss = (closs + ncloss) / 2.
            logging_out["inter_loss"] = inter_loss.data

            if self.consis_alpha > 0:        
                loss += self.consis_alpha * inter_loss
            
            
            mse_loss_fcn = torch.nn.MSELoss(reduction='sum')
            msel = mse_loss_fcn(F.logsigmoid(mix_logits), F.logsigmoid(t_logits.detach()))
            nmsel = mse_loss_fcn(F.logsigmoid(mix_neg_logits), F.logsigmoid(t_neg_logits.detach()))
            intra_loss = (msel + nmsel) / 2.
            logging_out["intra_loss"] = intra_loss.data

            if self.p_consis_alpha > 0:
                loss += self.p_consis_alpha * intra_loss
            
            pos_preds = torch.sigmoid(mix_logits)
            neg_preds = torch.sigmoid(mix_neg_logits)
        
            logging_out["ncorrect"] = (pos_preds >= 0.5).sum() + (neg_preds < 0.5).sum()
            logging_out["pos_acc"] = (pos_preds >= 0.5).sum() 
            logging_out["neg_acc"] = (neg_preds < 0.5).sum()

        else:
            loss = (- F.logsigmoid(pos_logits).mean() - F.logsigmoid(-neg_logits).mean()) / 2. 
            loss += (- F.logsigmoid(pos_logits_deltaH).mean() - F.logsigmoid(-neg_logits_deltaH).mean()) / 2. 
            loss += (- F.logsigmoid(pos_logits_deltaT).mean() - F.logsigmoid(-neg_logits_deltaT).mean()) / 2. 
            loss += (- F.logsigmoid(pos_logits_ori).mean() - F.logsigmoid(-neg_logits_ori).mean()) / 2. 
            
            loss = loss / 4.
            
            reg_loss = F.mse_loss(F.logsigmoid(pos_logits), F.logsigmoid(pos_logits_deltaH))
            reg_loss += F.mse_loss(F.logsigmoid(pos_logits), F.logsigmoid(pos_logits_deltaT))
            reg_loss += F.mse_loss(F.logsigmoid(pos_logits_deltaH), F.logsigmoid(pos_logits_deltaT))
            reg_loss += F.mse_loss(F.logsigmoid(pos_logits), F.logsigmoid(pos_logits_ori))
            
            reg_loss = reg_loss / 4.
            
            reg_loss_neg = F.mse_loss(F.logsigmoid(neg_logits), F.logsigmoid(neg_logits_deltaH))
            reg_loss_neg += F.mse_loss(F.logsigmoid(neg_logits), F.logsigmoid(neg_logits_deltaT))
            reg_loss_neg += F.mse_loss(F.logsigmoid(neg_logits_deltaH), F.logsigmoid(neg_logits_deltaT))
            reg_loss_neg += F.mse_loss(F.logsigmoid(neg_logits), F.logsigmoid(neg_logits_ori))

            reg_loss_neg = reg_loss_neg / 4.

            reg_loss += reg_loss_neg
            reg_loss = reg_loss / 2.
            
            # print(f"loss: {loss}, reg_loss: {reg_loss}")

            loss += self.reg_loss_weight * reg_loss

            pos_preds = torch.sigmoid(pos_logits)
            neg_preds = torch.sigmoid(neg_logits)
            
            logging_out = {
                "loss": loss.data,
                "ntokens": sample["ntokens"] * 2,
                "nsentences": sample_size * 2,
                "sample_size": sample_size * 2,
            }
            logging_out["ncorrect"] = (pos_preds >= 0.5).sum() + (neg_preds < 0.5).sum()
            logging_out["pos_acc"] = (pos_preds >= 0.5).sum() 
            logging_out["neg_acc"] = (neg_preds < 0.5).sum()

        return loss, sample_size, logging_out

    def forward_inference(self, model, sample, reduce=True):
        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        
        pos_input = self.build_input(sample, self.classification_head_name)
        pos_logits = model(**pos_input)
        neg_input = self.build_input(sample, self.classification_head_name, neg=True)
        neg_logits = model(**neg_input)
        
        preds = []
        
        if isinstance(pos_logits, tuple):
            pos_logits = pos_logits[0]
            neg_logits = neg_logits[0]
        
        pos_preds = torch.sigmoid(pos_logits.squeeze().float()).detach().cpu().numpy()
        neg_preds = torch.sigmoid(neg_logits.squeeze().float()).detach().cpu().numpy()
        preds.append(pos_preds)
        preds.append(neg_preds)
        
        targets = []
        pos_target = torch.ones(len(pos_preds))
        neg_target = torch.zeros(len(neg_preds))
        targets.append(pos_target)
        targets.append(neg_target)

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        return preds, targets, sample['target'].detach().cpu().numpy()

    def tsne_inference(self, model, sample, reduce=True):
        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        
        pos_input = self.build_input(sample, self.classification_head_name)
        a_embed, b_embed  = model.forward_embed(**pos_input)
        return a_embed, b_embed


    @staticmethod
    def reduce_metrics(logging_outputs):

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1)
            
            pos_acc = sum(log.get("pos_acc", 0) for log in logging_outputs)
            metrics.log_scalar("pos_acc", 100.0 * 2 * pos_acc / nsentences, nsentences, round=1)
            neg_acc = sum(log.get("neg_acc", 0) for log in logging_outputs)
            metrics.log_scalar("neg_acc", 100.0 * 2 * neg_acc / nsentences, nsentences, round=1)

        if len(logging_outputs) > 0 and "inter_loss" in logging_outputs[0]:
            inter_loss_sum = sum(log.get("inter_loss", 0) for log in logging_outputs)
            metrics.log_scalar("inter_loss", inter_loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "intra_loss" in logging_outputs[0]:
            intra_loss_sum = sum(log.get("intra_loss", 0) for log in logging_outputs)
            metrics.log_scalar("intra_loss", intra_loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "t_ncorrect" in logging_outputs[0]:
            t_ncorrect = sum(log.get("t_ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("t_accuracy", 100.0 * t_ncorrect / nsentences, nsentences, round=1)
            
            pos_acc = sum(log.get("t_pos_acc", 0) for log in logging_outputs)
            metrics.log_scalar("t_pos_acc", 100.0 * 2 * pos_acc / nsentences, nsentences, round=1)
            neg_acc = sum(log.get("t_neg_acc", 0) for log in logging_outputs)
            metrics.log_scalar("t_neg_acc", 100.0 * 2 * neg_acc / nsentences, nsentences, round=1)

        if len(logging_outputs) > 0 and "g_ncorrect" in logging_outputs[0]:
            g_ncorrect = sum(log.get("g_ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("g_accuracy", 100.0 * g_ncorrect / nsentences, nsentences, round=1)
            
            pos_acc = sum(log.get("g_pos_acc", 0) for log in logging_outputs)
            metrics.log_scalar("g_pos_acc", 100.0 * 2 * pos_acc / nsentences, nsentences, round=1)
            neg_acc = sum(log.get("g_neg_acc", 0) for log in logging_outputs)
            metrics.log_scalar("g_neg_acc", 100.0 * 2 * neg_acc / nsentences, nsentences, round=1)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
