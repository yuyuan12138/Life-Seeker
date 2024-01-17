import shutil
import numpy as np
import torch
import torch


class Metrics:
    def __init__(self, data_label, data_predict):
        self.data_label = data_label
        self.data_predict = data_predict
        self._cal_tp_fn_fp_tn_()
        self._cal_acc_()
        self._cal_sn_()
        self._cal_p_()
        self._cal_sp_()
        self._cal_mcc_()
        self._cal_f1_()

    def get_all(self):
        return [self.get_acc(), self.get_sn(), self.get_sp(), self.get_p(), self.get_mcc(), self.get_f1()]

    def get_acc(self):
        return self.acc

    def get_sn(self):
        return self.sn

    def get_sp(self):
        return self.sp

    def get_p(self):
        return self.p

    def get_mcc(self):
        return self.mcc

    def get_f1(self):
        return self.f1

    def _cal_tp_fn_fp_tn_(self):
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0
        for idx, value in enumerate(self.data_label):
            if value == 1:
                if self.data_predict[idx] == 1:
                    self.tp += 1
                elif self.data_predict[idx] == 0:
                    self.fn += 1
            elif value == 0:
                if self.data_predict[idx] == 1:
                    self.fp += 1
                elif self.data_predict[idx] == 0:
                    self.tn += 1

    def _cal_acc_(self):
        self.acc = (self.tn + self.tp) / (self.tn + self.tp + self.fp + self.fn)

    def _cal_sn_(self):
        self.sn = self.tp / (self.fn + self.tp)

    def _cal_sp_(self):
        self.sp = self.tn / (self.tn + self.fp)

    def _cal_p_(self):
        self.p = self.tp / (self.tp + self.fp)

    def _cal_mcc_(self):
        self.mcc = (self.tn * self.tp - self.fn * self.fp) / np.sqrt((self.tn + self.fp) * (self.fn + self.tp) *
                                                                     (self.tn + self.fn) * (self.tp + self.fp))

    def _cal_f1_(self):
        self.f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn)


class Save_Load_Checkpoint:
    def __init__(self, save_path, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
        self.save_path = save_path
        self.filename = filename
        self.best_filename = best_filename

    def save_checkpoint(self, model, is_best):
        print("======> Save checkpoint")
        torch.save(model, self.save_path + self.filename)
        if is_best:
            shutil.copyfile(self.save_path + self.filename, self.save_path + self.best_filename)

    def load_checkpoint(self):
        print("======> Load checkpoint")
        model = torch.load(self.save_path + self.best_filename)
        return model
