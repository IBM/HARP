#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

import os
import joblib
import pandas as pd
import pickle

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

class PMCA_pred_models:
    def __init__(self, M=0, N=0, K=0, gemm_fast=False):
        self.M = M
        self.N = N
        self.K = K
        self.gemm_fast = gemm_fast

    # Operations for different models
    def tot_num_ops_gemm(self, M,N,K):
        return M*N*(K+(K-1))+M*N

    def tot_num_ops_softmax(self,M,N):
        return M*(N-1+3*N+N-1+N+1)

    def tot_num_ops_LayerNorm(self,M,N):
        return M*(N+3*N-1+2*(N+1))

    def tot_num_ops_GeLU(self,M,N):
        return 10*M*N

    def tot_num_ops_PartialSoftmax(self, M,N,K):
        return 4*(M*K)+4*M+(K-1)*M
    def tot_num_ops_PartialScale(self, M,N,K):
        return M*N
    def tot_num_ops_FinalScale(self, M,N,K):
        return M*N
    
    def tot_num_ops_ElementwiseSum(self,M,N):
        return M*N

    # Convert input to dictionary (DataFrame)
    def convert_inp_into_dic(self, model_type=None):
        if model_type == "GEMM":
            tot_num_ops = self.tot_num_ops_gemm(self.M, self.N, self.K)
            X = pd.DataFrame({
                "M": [self.M],
                "N": [self.N],
                "K": [self.K],
                "Tot n ops": [tot_num_ops]
            })
        elif model_type == "Softmax":
            tot_num_ops = self.tot_num_ops_softmax(self.M, self.N)
            X = pd.DataFrame({
                "M": [self.M],
                "N": [self.N],
                "Tot n ops": [tot_num_ops]
            })
        elif model_type == "LayerNorm":
            tot_num_ops = self.tot_num_ops_LayerNorm(self.M, self.N)
            X = pd.DataFrame({
                "M": [self.M],
                "N": [self.N],
                "Tot n ops": [tot_num_ops]
            })
        elif model_type == "GeLU":
            tot_num_ops = self.tot_num_ops_GeLU(self.M, self.N)
            X = pd.DataFrame({
                "M": [self.M],
                "N": [self.N],
                "Tot n ops": [tot_num_ops]
            })
        elif model_type == "PartialSoftmax":
            tot_num_ops = self.tot_num_ops_PartialSoftmax(self.M, self.N, self.K)
            X = pd.DataFrame({
                "Br": [self.M],
                "d": [self.N],
                "Bc": [self.K],
                "Tot n ops": [tot_num_ops]
            })
        elif model_type == "PartialScale":
            tot_num_ops = self.tot_num_ops_PartialScale(self.M, self.N, self.K)
            X = pd.DataFrame({
                "Br": [self.M],
                "d": [self.N],
                "Bc": [self.K],
                "Tot n ops": [tot_num_ops]
            })
        elif model_type == "FinalScale":
            tot_num_ops = self.tot_num_ops_FinalScale(self.M, self.N, self.K)
            X = pd.DataFrame({
                "Br": [self.M],
                "d": [self.N],
                "Bc": [self.K],
                "Tot n ops": [tot_num_ops]
            })
        elif model_type == "ElementwiseSum":
            tot_num_ops = self.tot_num_ops_ElementwiseSum(self.M, self.N)
            X = pd.DataFrame({
                "M": [self.M],
                "K": [self.N],
                "Tot n ops": [tot_num_ops]
            })
        else:
            raise ValueError("Invalid model type")

        return X

    # Load prediction models
    def load_models(self):
        if self.gemm_fast:
            # PMCA_RED (with GEMM accelerator)
            self.GEMM_pred_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model_GEMM_red.pkl'))
        else:
            # PMCA (no GEMM accelerator, slower GEMM)
            self.GEMM_pred_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model_GEMM.pkl'))

        self.Softmax_pred_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model_softmax.pkl'))
        self.LayerNorm_pred_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model_LayerNorm.pkl'))
        self.GeLU_pred_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model_GeLU.pkl'))
        self.PartialSoftmax_pred_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model_partial_softmax.pkl'))
        self.PartialScale_pred_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model_partial_scale.pkl'))
        self.FinalScale_pred_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model_final_scale.pkl'))
        self.ElementwiseSum_pred_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model_ElementwiseSum.pkl'))
        
    # Predict using the appropriate model
    def predict(self, model_type="GEMM"):
        # Load models if not already loaded
        self.load_models()
        
        if model_type == "GEMM":
            X = self.convert_inp_into_dic(model_type="GEMM")
            prediction = self.GEMM_pred_model.predict(X)
        elif model_type == "Softmax":
            X = self.convert_inp_into_dic(model_type="Softmax")
            prediction = self.Softmax_pred_model.predict(X)
        elif model_type == "LayerNorm":
            X = self.convert_inp_into_dic(model_type="LayerNorm")
            prediction = self.LayerNorm_pred_model.predict(X)
        elif model_type == "GeLU":
            X = self.convert_inp_into_dic(model_type="GeLU")
            prediction = self.GeLU_pred_model.predict(X)
        elif model_type == "PartialSoftmax":
            X = self.convert_inp_into_dic(model_type="PartialSoftmax")
            prediction = self.PartialSoftmax_pred_model.predict(X)
        elif model_type == "PartialScale":
            X = self.convert_inp_into_dic(model_type="PartialScale")
            prediction = self.PartialScale_pred_model.predict(X)
        elif model_type == "FinalScale":
            X = self.convert_inp_into_dic(model_type="FinalScale")
            prediction = self.FinalScale_pred_model.predict(X)
        elif model_type == "ElementwiseSum":
            X = self.convert_inp_into_dic(model_type="ElementwiseSum")
            prediction = self.ElementwiseSum_pred_model.predict(X)
        else:
            raise ValueError("Invalid model type")

        return prediction