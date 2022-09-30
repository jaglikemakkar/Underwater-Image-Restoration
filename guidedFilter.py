# Reference : https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2019.1471

import numpy as np
import cv2

class GuidedFilter:
    def __init__(self, img, r, eps):

        # Changing data type of image to float
        if img.dtype != np.float32:
            img = (1.0 / 255.0) * np.float32(img)

        self.img = img
        self.r = 2*r + 1
        self.eps = eps
        self.begin()
    
    # Computing mean of input image over box of size (r,r)
    def mean(self, img):
        r = self.r
        return cv2.blur(img, (r,r))
        

    def begin(self):
        img = self.img
        eps = self.eps

        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]
        
        # Calculating mean of R,G,B over box of size (r,r)
        img_r_mean = self.mean(img_r)
        img_g_mean = self.mean(img_g)
        img_b_mean = self.mean(img_b)

        # Calculating variance
        # var = mean(img * img) - (img_mean * img_mean)
        img_rr_var = self.mean(img_r * img_r) - (img_r_mean * img_r_mean) + eps
        img_rg_var = self.mean(img_r * img_g) - (img_r_mean * img_g_mean)
        img_rb_var = self.mean(img_r * img_b) - (img_r_mean * img_b_mean)
        img_gg_var = self.mean(img_g * img_g) - (img_g_mean * img_g_mean) + eps
        img_gb_var = self.mean(img_g * img_b) - (img_g_mean * img_b_mean)
        img_bb_var = self.mean(img_b * img_b) - (img_b_mean * img_b_mean) + eps

        img_rr_inv = (img_gg_var * img_bb_var) - (img_gb_var)**2
        img_rg_inv = (img_gb_var * img_rb_var) - (img_rg_var * img_bb_var)
        img_rb_inv = (img_rg_var * img_gb_var) - (img_gg_var * img_rb_var)
        img_gg_inv = (img_rr_var * img_bb_var) - (img_rb_var)**2
        img_gb_inv = (img_rb_var * img_rg_var) - (img_rr_var * img_gb_var)
        img_bb_inv = (img_rr_var * img_gg_var) - (img_rg_var)**2

        img_cov = (img_rr_inv * img_rr_var) + (img_rg_inv * img_rg_var) + (img_rb_inv * img_rb_var)

        img_rr_inv = img_rr_inv / img_cov
        img_rg_inv = img_rg_inv / img_cov
        img_rb_inv = img_rb_inv / img_cov
        img_gg_inv = img_gg_inv / img_cov
        img_gb_inv = img_gb_inv / img_cov
        img_bb_inv = img_bb_inv / img_cov

        self.img_r_mean = img_r_mean
        self.img_g_mean = img_g_mean
        self.img_b_mean = img_b_mean

        self.img_rr_inv = img_rr_inv
        self.img_rg_inv = img_rg_inv
        self.img_rb_inv = img_rb_inv
        self.img_gg_inv = img_gg_inv
        self.img_gb_inv = img_gb_inv
        self.img_bb_inv = img_bb_inv

    def getCoefficients(self, p):
        img = self.img
        r = self.r
        
        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]

        img_r_mean = self.img_r_mean
        img_g_mean = self.img_g_mean
        img_b_mean = self.img_b_mean

        # Calculating mean
        p_mean = self.mean(p)

        # Calculating mean of I_i * p_i
        img_pr_mean = self.mean(img_r * p)
        img_pg_mean = self.mean(img_g * p)
        img_pb_mean = self.mean(img_b * p)

        # Calculating numerator = (I_i * p_i) - (u_k * p'_k)
        img_pr_cov = img_pr_mean - img_r_mean * p_mean
        img_pg_cov = img_pg_mean - img_g_mean * p_mean
        img_pb_cov = img_pb_mean - img_b_mean * p_mean

        # Calculating coefficient a_k = (1/w) ((I_i * p_i) - (u_k * p'_k)) / (sigma**2 + eps)
        a_r = (self.img_rr_inv * img_pr_cov) + (self.img_rg_inv * img_pg_cov) + (self.img_rb_inv * img_pb_cov)
        a_g = (self.img_rg_inv * img_pr_cov) + (self.img_gg_inv * img_pg_cov) + (self.img_gb_inv * img_pb_cov)
        a_b = (self.img_rb_inv * img_pr_cov) + (self.img_gb_inv * img_pg_cov) + (self.img_bb_inv * img_pb_cov)

        # Calculating coefficient b_k = (p'_k - a_k * u_k)
        b = p_mean - (a_r * img_r_mean + a_g * img_g_mean + a_b * img_b_mean)

        # Calculating mean of coefficients
        a_r_mean = self.mean(a_r)
        a_g_mean = self.mean(a_g)
        a_b_mean = self.mean(a_b)
        b_mean = self.mean(b)

        return a_r_mean, a_g_mean, a_b_mean, b_mean
    
    def filter(self, p):
        # Converting image to float
        # if p.dtype != np.float32:
        #     p = np.float32(p)
        
        img = self.img

        # Computing coefficients a_k & b_k
        a_r_mean, a_g_mean, a_b_mean, b_mean = self.getCoefficients(p)

        # print("Coefficients : ", a_r_mean, a_g_mean, a_b_mean, b_mean)

        img_r, img_g, img_b = img[:,:,0], img[:,:,1], img[:,:,2]

        # Finding ouput image q = a_k * I_i + b_k
        out_img = (a_r_mean * img_r) + (a_g_mean * img_g) + (a_b_mean * img_b) + b_mean 

        return out_img
        