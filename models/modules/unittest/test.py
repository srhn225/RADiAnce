import torch
import torch.nn as nn
import unittest
from models.modules.EPT.ept import *

class TestEPTLayer(unittest.TestCase):
    def test_forward_pass(self):
        B, N, P, D, D_FFN, H_heads = 2, 100, 10, 512, 512, 4
        H = torch.randn(B, N, D)
        V = torch.randn(B, N, 3, D)
        prompt = torch.randn(B* P, D)
        D_batch = torch.randn(B, N, N)
        rbf_feat = torch.randn(6,B, 4,N, N)
        H_mask = torch.ones(B, N)

        model = EPTLayerInContextConditioning(D, D_FFN, H_heads, layer_idx=2)
        out_H, out_V = model(H, V, cached_info=(D_batch, rbf_feat, H_mask), prompt_feature=prompt)

        self.assertEqual(out_H.shape, (B, N, D))
        self.assertEqual(out_V.shape, (B, N, 3, D))
class TestEPTLayerAdaLNZero(unittest.TestCase):
    def test_forward_pass_adaln_zero(self):
        # Setup
        B, N, P, D, D_FFN, H_heads = 2, 100, 10, 512, 512, 4
        H = torch.randn(B, N, D)
        V = torch.randn(B, N, 3, D)
        prompt = torch.randn(B* P, D)
        D_batch = torch.randn(B, N, N)
        rbf_feat = torch.randn(6, B, 4, N, N)
        H_mask = torch.ones(B, N)

        model = EPTLayerAdaLNZero(
            d_hidden=D,
            d_ffn=D_FFN,
            n_heads=H_heads,
            layer_idx=2,
            act_fn=nn.SiLU(),
            layer_norm='pre',
            residual=True,
            efficient=False,
            vector_act='none',
            attn_bias=True,
        )

        # Run
        out_H, out_V = model(H, V, cached_info=(D_batch, rbf_feat, H_mask), prompt_feature=prompt)

        # Assert
        self.assertEqual(out_H.shape, (B, N, D))
        self.assertEqual(out_V.shape, (B, N, 3, D))
class TestEPTLayerrag(unittest.TestCase):
    def test_forward_pass_rag(self):
        # Setup
        B, N, P, D, D_FFN, H_heads = 2, 100, 10, 512, 512, 4
        H = torch.randn(B, N, D)
        V = torch.randn(B, N, 3, D)
        prompt = torch.randn(B*P, D)
        D_batch = torch.randn(B, N, N)
        rbf_feat = torch.randn(6, B, 4, N, N)
        H_mask = torch.ones(B, N)

        model = EPTLayerrag(
            d_hidden=D,
            d_ffn=D_FFN,
            n_heads=H_heads,
            layer_idx=2,
            act_fn=nn.SiLU(),
            layer_norm='pre',
            residual=True,
            efficient=False,
            vector_act='none',
            attn_bias=True,
        )

        # Run
        out_H, out_V = model(H, V, cached_info=(D_batch, rbf_feat, H_mask), prompt_feature=prompt)

        # Assert
        self.assertEqual(out_H.shape, (B, N, D))
        self.assertEqual(out_V.shape, (B, N, 3, D))
class TestEPTLayerAdaLNAttn(unittest.TestCase):
    def test_forward_pass_AdaLNAttn(self):
        # Setup
        B, N, P, D, D_FFN, H_heads = 2, 100, 10, 512, 512, 4
        H = torch.randn(B, N, D)
        V = torch.randn(B, N, 3, D)
        prompt = torch.randn(B*P, D)
        D_batch = torch.randn(B, N, N)
        rbf_feat = torch.randn(6, B, 4, N, N)
        H_mask = torch.ones(B, N)

        model = EPTLayerAdaLNAttn(
            d_hidden=D,
            d_ffn=D_FFN,
            n_heads=H_heads,
            layer_idx=2,
            act_fn=nn.SiLU(),
            layer_norm='pre',
            residual=True,
            efficient=False,
            vector_act='none',
            attn_bias=True,
        )

        # Run
        out_H, out_V = model(H, V, cached_info=(D_batch, rbf_feat, H_mask), prompt_feature=prompt)

        # Assert
        self.assertEqual(out_H.shape, (B, N, D))
        self.assertEqual(out_V.shape, (B, N, 3, D))
# unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestEPTLayerAdaLNZero))
if __name__ == "__main__":
    unittest.main()
