# This code is built from the Huggingface repository: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py, and
# https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
# Copyright 2022- The Hugging Face team. All rights reserved.
#                               Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
# ==========================================================================================
#
# modifications are MIT License. To view a copy of the license, visit MIT_LICENSE.md.
#
# ==========================================================================================
#                               Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

#    1. Definitions.

#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.

#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.

#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.

#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.

#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.

#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.

#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).

#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.

#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."

#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.

#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.

#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.

#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:

#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and

#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and

#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and

#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.

#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.

#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.

#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.

#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.

#    END OF TERMS AND CONDITIONS

#    APPENDIX: How to apply the Apache License to your work.

#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.

#    Copyright [yyyy] [name of copyright owner]

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Callable, Optional
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from accelerate.logging import get_logger
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline, StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.models.attention import Attention
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.utils import replace_example_docstring
import torch.distributions as dist

from sklearn.cluster import SpectralClustering
import seaborn as sns
import collections
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import os
import matplotlib as mpl
import torchvision
import torch.nn.functional as F
from PIL import Image
from einops import rearrange


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

logger = get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

def set_use_memory_efficient_attention_xformers(
    self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
):
    if use_memory_efficient_attention_xformers:
        if self.added_kv_proj_dim is not None:
            # TODO(Anton, Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
            # which uses this type of cross attention ONLY because the attention mask of format
            # [0, ..., -10.000, ..., 0, ...,] is not supported
            raise NotImplementedError(
                "Memory efficient attention with `xformers` is currently not supported when"
                " `self.added_kv_proj_dim` is defined."
            )
        elif not is_xformers_available():
            raise ModuleNotFoundError(
                (
                    "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                    " xformers"
                ),
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                " only available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e

        processor = CustomizationXFormersAttnProcessor(attention_op=attention_op)
    else:
        processor = CustomizationAttnProcessor()

    self.set_processor(processor)

def find_sublist_indices(long_list, short_list):
    len_long = len(long_list)
    len_short = len(short_list)

    for i in range(len_long - len_short + 1):
        if long_list[i:i + len_short] == short_list:
            return i, i + len_short - 1 
    return -1, -1  

def _symmetric_kl(attention_map1, attention_map2):
    attention_map1 = attention_map1.reshape(int(attention_map1.shape[0]**0.5), int(attention_map1.shape[0]**0.5))
    attention_map2 = attention_map2.reshape(int(attention_map2.shape[0]**0.5), int(attention_map2.shape[0]**0.5))

    n = 4
    attention_map1 = attention_map1.unfold(0, n, n).unfold(1, n, n)
    #attention_map1 = attention_map1.max(dim=2).values.max(dim=2).values
    attention_map1 = attention_map1.mean(dim=2).mean(dim=2)

    attention_map2 = attention_map2.unfold(0, n, n).unfold(1, n, n)
    #attention_map2 = attention_map2.max(dim=2).values.max(dim=2).values
    attention_map2 = attention_map2.mean(dim=2).mean(dim=2)

    if len(attention_map1.shape) > 1:
        attention_map1 = attention_map1.reshape(-1)
    if len(attention_map2.shape) > 1:
        attention_map2 = attention_map2.reshape(-1)

    attention_map1 = F.softmax(attention_map1.float()*100, dim=0)
    attention_map2 = F.softmax(attention_map2.float()*100, dim=0)

    p = dist.Categorical(probs=attention_map1)
    q = dist.Categorical(probs=attention_map2)

    kl_divergence_pq = dist.kl_divergence(p, q)
    kl_divergence_qp = dist.kl_divergence(q, p)

    avg_kl_divergence = (kl_divergence_pq + kl_divergence_qp) / 2
    return avg_kl_divergence

def _calculate_negative_loss(attention_maps, src_indices, nagative_indices):
    negative_loss = []
    computed_pairs = set()

    for negative_idx in nagative_indices:
        wp_neg_loss = []
        for t in src_indices:
            pair_key = (t, negative_idx)
            if pair_key not in computed_pairs:
                wp_neg_loss.append(
                    _symmetric_kl(
                        attention_maps[:, t], attention_maps[:, negative_idx]
                    )
                )
                computed_pairs.add(pair_key)

        negative_loss.append(max(wp_neg_loss) if wp_neg_loss else 0)

    return negative_loss

# AttnProcessor for fine-tuning
class CustomizationAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if isinstance(encoder_hidden_states, dict):
            pos_info = encoder_hidden_states['pos_info']
            encoder_hidden_states = encoder_hidden_states['prompt_embeds_input']

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if crossattn:
            encoder_hidden_states_replace = encoder_hidden_states[-1].unsqueeze(0)  
            encoder_hidden_states = encoder_hidden_states[:hidden_states.shape[0]]
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            key_replace = attn.to_1k(encoder_hidden_states_replace)
            value_replace = attn.to_1v(encoder_hidden_states_replace)
            
            key_p1 = key_replace[0, 0:pos_info["replace_start"], :].unsqueeze(0)
            key_p2 = key[0, pos_info["ins_start"]:pos_info["ins_end"], :].unsqueeze(0)
            key_p3 = key_replace[0, pos_info["replace_end"]:, :].unsqueeze(0)
            key[0] = torch.cat([key_p1, key_p2, key_p3], dim=1)

            value_p1 = value_replace[0, 0:pos_info["replace_start"], :].unsqueeze(0)
            value_p2 = value[0, pos_info["ins_start"]:pos_info["ins_end"], :].unsqueeze(0)
            value_p3 = value_replace[0, pos_info["replace_end"]:, :].unsqueeze(0)
            value[0] = torch.cat([value_p1, value_p2, value_p3], dim=1)

        else:
            key = attn.to_k(encoder_hidden_states)  # torch.Size([bs, 1024, 640])
            value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :]*0.
            key = detach*key + (1-detach)*key.detach()
            value = detach*value + (1-detach)*value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

save_weight = {}

# AttnProcessor for inference
class MultiConceptprocessor:
    def __init__(
        self,
        name=None,
        step=None,
        save_attn=False,
        refine_attn=False,
        refine_info=None,
        attn_dict=None,
        concept_list=None,
        concept_bank=None,
        inf_batch_size=1,
        ):
        super().__init__()
        self.name = name
        self.step = step
        self.save_attn = save_attn
        self.refine_attn = refine_attn
        self.refine_info = refine_info
        self.attn_dict = attn_dict
        self.concept_list = concept_list
        self.concept_bank = concept_bank
        self.negative_loss = 0.0
        self.positive_loss = 0.0
        self.inf_batch_size = inf_batch_size

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ): 
        # hidden_states  image_feature 4 /1024/4096
        # encoder_hidden_states  textual_feature. [6, 77, 2048]
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape
        _, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        sample_number = int(batch_size/2)

        if encoder_hidden_states is None:
            # self_attention
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        concept_number = self.concept_list['concept_num']
        if crossattn:
            key = attn.to_k(encoder_hidden_states, scale=scale)
            value = attn.to_v(encoder_hidden_states, scale=scale)

            if concept_number==1 and self.concept_list['get_concept_bank']:
                # extract single concept representation
                start = self.concept_list['concept1']['start']
                end = self.concept_list['concept1']['end']

                encoder_hidden_states_plain = encoder_hidden_states[:batch_size]
                encoder_hidden_states_1 = encoder_hidden_states[batch_size:batch_size+sample_number]

                key = attn.to_k(encoder_hidden_states_plain, scale=scale)
                value = attn.to_v(encoder_hidden_states_plain, scale=scale)
                key_1 = attn.to_k1(encoder_hidden_states_1, scale=scale)
                value_1 = attn.to_v1(encoder_hidden_states_1, scale=scale)

                save_weight[self.name] = {}
                save_weight[self.name]['k'] = key_1[0, start:end, :]
                save_weight[self.name]['v'] = value_1[0, start:end, :]
                    
                key[sample_number:, start:end, :] = key_1[:, start:end, :]
                value[sample_number:, start:end, :] = value_1[:, start:end, :]

            else:
                # multi-concept blending
                if not self.save_attn and self.step/1000<self.concept_list['blending_start']:
                    for i in range(1, concept_number + 1):
                        concept_name = f'concept{i}'
                        concept = self.concept_bank[concept_name][self.name]
                        start = self.concept_list[concept_name]['start']
                        end = self.concept_list[concept_name]['end']

                        key[sample_number:, start:end, :] = concept['k'].unsqueeze(0)
                        value[sample_number:, start:end, :] = concept['v'].unsqueeze(0)
        else:
            # self attention
            key = attn.to_k(encoder_hidden_states, scale=scale)
            value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        #attention_probs = attn.get_attention_scores(query, key, attention_mask)
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input

        if crossattn and self.concept_list['get_concept_bank'] != 1 and self.inf_batch_size == batch_size:
            bind_dict = self.concept_list['bind_dict']
            for bind_key in bind_dict.keys():
                int_key = int(bind_key)
                attention_scores[:, :, int_key] = attention_scores[:, :, bind_dict[bind_key]]

        ########### refine attention maps
        if self.refine_attn and crossattn:
            sizereg = 1
            if 'creg' in self.concept_list: 
                creg = self.concept_list['creg']
            else:
                creg = 2

            res_key = attention_scores.shape[-2]
            cross_attention_weight = self.refine_info['creg_mask'][res_key]
            cross_attention_weight_obj = self.refine_info['creg_mask_obj'][res_key]

            batch_size, seq_len, dim = attention_scores.shape
            attention_scores_now = attention_scores.reshape(batch_size // attn.heads, attn.heads, seq_len, dim)
            attention_scores_now = attention_scores[int(attention_scores.shape[0]/2):].reshape(-1, seq_len, dim)

            bs, seq_l, wdim = cross_attention_weight.shape
            cross_attention_weight = cross_attention_weight.unsqueeze(dim=1).repeat(1, attn.heads, 1, 1).reshape(-1, seq_l, wdim)
            cross_attention_weight_obj = cross_attention_weight_obj.unsqueeze(dim=1).repeat(1, attn.heads, 1, 1).reshape(-1, seq_l, wdim)

            layouts_s = cross_attention_weight.sum(1).unsqueeze(dim=1)
            size_reg = (1 - sizereg * layouts_s / res_key)    # 60, 1, 77]
            
            layouts_obj = cross_attention_weight_obj.sum(1).unsqueeze(dim=1)
            size_reg_obj=(1 - sizereg * layouts_obj / res_key) 

            min_value=attention_scores_now.min(-1)[0].unsqueeze(-1)   # [60, 4096, 1]
            max_value=attention_scores_now.max(-1)[0].unsqueeze(-1)   # [60, 4096, 1]
            treg=torch.pow(self.step/1000,5).to(attention_scores.device)  # tensor(781.) , attention_scores: [60, 4096, 77], cross_attention_weight: [60, 4096, 77],

            attention_scores[attention_scores.shape[0]//2:] = attention_scores_now + (cross_attention_weight_obj> 0) * size_reg_obj * creg * treg * (max_value - attention_scores_now)
            attention_scores[attention_scores.shape[0]//2:] = attention_scores_now - (cross_attention_weight> 0) * size_reg * creg * treg * (attention_scores_now - min_value)

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()
        
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        # calculate loss
        if crossattn and self.concept_list['blending_guidance'] and self.inf_batch_size != batch_size:
            for concept_index in range(0, concept_number):
                concept_name = f'concept{concept_index+1}'

                # positive loss
                noun_list = self.concept_list['noun_list'][concept_index]
                modifier_index = noun_list[0] - 1

                attention_maps = attention_probs.mean(dim=0)

                wp_pos_loss = [
                    _symmetric_kl(attention_maps[:, modifier_index], attention_maps[:, d])
                    for d in noun_list
                ]
                self.positive_loss = max(wp_pos_loss)

                # negative loss
                for re_index in range(0, concept_number):
                    if re_index == concept_index:
                        continue

                    re_noun_list = self.concept_list['noun_list'][re_index]
                    negative_loss = _calculate_negative_loss(
                            attention_maps, [modifier_index] + noun_list, [re_noun_list[0]-1] + re_noun_list
                        )
                    self.negative_loss += -sum(negative_loss) / len([re_noun_list[0]-1] + re_noun_list)

        ### save attn
        if self.save_attn==True:
            self_attn_dict=collections.defaultdict(list)
            cross_attn_dict=collections.defaultdict(list)

            head_size = attn.heads
            if "attn2" in self.name:
                batch_size, seq_len, dim = attention_probs.shape
                attention_probs_save = attention_probs.reshape(batch_size // head_size, head_size, seq_len, dim).mean(dim=1)
                attention_probs_save = attention_probs_save[int(attention_probs_save.shape[0]/2):]
                cross_attn_dict[self.name]=attention_probs_save
                self.attn_dict['cross_attn'][self.name] = attention_probs_save
            elif "attn1" in self.name and attention_probs.shape[2]==4096:  ###1024/4096
                batch_size, seq_len, dim = attention_probs.shape
                attention_probs_save = attention_probs.reshape(batch_size // head_size, head_size, seq_len, dim).mean(dim=1)
                attention_probs_save = attention_probs_save[int(attention_probs_save.shape[0]/2):]

                self_attn_dict[self.name]=attention_probs_save
                self.attn_dict['self_attn'][self.name]=attention_probs_save
        
        hidden_states = torch.bmm(attention_probs, value)  # hidden_states.shape [60, 4096, 64], [120, 1024, 64]
        hidden_states = attn.batch_to_head_dim(hidden_states) # hidden_states.shape [6, 4096, 640], [6, 1024, 1280], 

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CustomizationXFormersAttnProcessor:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.cross_attention_norm:
                encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :]*0.
            key = detach*key + (1-detach)*key.detach()
            value = detach*value + (1-detach)*value.detach()

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class LaTexBlendXLPipeline(StableDiffusionXLPipeline):
    r"""
    Pipeline for custom diffusion model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
        modifier_token: list of new modifier tokens added or to be added to text_encoder
        modifier_token_id: list of id of new modifier tokens added or to be added to text_encoder
        modifier_token_id_2: list of id of new modifier tokens added or to be added to text_encoder_2
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        modifier_token: list = [],
        modifier_token_id: list = [],
        modifier_token_id_2: list = [],
        stage: Optional[str] = "inference",
        get_concept_bank: int=0,

    ):
        super().__init__(vae,
                         text_encoder,
                         text_encoder_2,
                         tokenizer,
                         tokenizer_2,
                         unet,
                         scheduler,
                         force_zeros_for_empty_prompt,
                         add_watermarker,
                         )

        # change attn class
        self.modifier_token = modifier_token
        self.modifier_token_id = modifier_token_id
        self.modifier_token_id_2 = modifier_token_id_2

        def fn_recursive_attn_alter(name: str, module: torch.nn.Module):
            if hasattr(module, "set_processor"):
                if 'attn2' in name:
                    setattr(module, 'to_k1', LoRACompatibleLinear(module.cross_attention_dim, module.inner_dim, bias=False).half())
                    setattr(module, 'to_v1', LoRACompatibleLinear(module.cross_attention_dim, module.inner_dim, bias=False).half())

            for sub_name, child in module.named_children():
                fn_recursive_attn_alter(f"{name}.{sub_name}", child)

        if stage == 'inference' and get_concept_bank:
            for name, module in self.unet.named_children():
                fn_recursive_attn_alter(name, module)

    def save_pretrained(self, save_path, freeze_model="crossattn_kv", save_text_encoder=False, all=False):
        if all:
            super().save_pretrained(save_path)
        else:
            delta_dict = {'unet': {}, 'modifier_token': {}}
            if self.modifier_token is not None:
                print(self.modifier_token_id, self.modifier_token)
                for i in range(len(self.modifier_token_id)):
                    delta_dict['modifier_token'][self.modifier_token[i]] = []
                    learned_embeds = self.text_encoder.get_input_embeddings().weight[self.modifier_token_id[i]]
                    learned_embeds_2 = self.text_encoder_2.get_input_embeddings().weight[self.modifier_token_id_2[i]]
                    delta_dict['modifier_token'][self.modifier_token[i]].append(learned_embeds.detach().cpu())
                    delta_dict['modifier_token'][self.modifier_token[i]].append(learned_embeds_2.detach().cpu())
            if save_text_encoder:
                delta_dict['text_encoder'] = self.text_encoder.state_dict()
                delta_dict['text_encoder_2'] = self.text_encoder_2.state_dict()
            for name, params in self.unet.named_parameters():
                if freeze_model == "crossattn":
                    if 'attn2' in name:
                        delta_dict['unet'][name] = params.cpu().clone()
                elif freeze_model == "crossattn_kv":
                    if 'attn2.to_k' in name or 'attn2.to_v' in name:
                        delta_dict['unet'][name] = params.cpu().clone()
                else:
                    raise ValueError(
                            "freeze_model argument only supports crossattn_kv or crossattn"
                        )
            torch.save(delta_dict, save_path)

    def load_model(self, concept_list, compress=False, deep_replace=False):
        modifier_tokens = concept_list['modifier_tokens']

        if '+' in modifier_tokens:
            modifier_tokens = modifier_tokens.split('+')
        else:
            modifier_tokens = [modifier_tokens]
        
        for modifier_token in modifier_tokens:
            print('modifier_token', modifier_token)
            num_added_tokens = self.tokenizer.add_tokens(modifier_token)
            num_added_tokens_2 = self.tokenizer_2.add_tokens(modifier_token)

        if "concept1" in concept_list and  concept_list["get_concept_bank"]:
            st = torch.load(concept_list['concept1']["delta_ckpt"])
            if 'text_encoder' in st:
                self.text_encoder.load_state_dict(st['text_encoder'])
                self.text_encoder_2.load_state_dict(st['text_encoder_2'])
            if 'modifier_token' in st:
                modifier_tokens_st = list(st['modifier_token'].keys())

                id_1 = self.tokenizer.convert_tokens_to_ids(modifier_tokens[0])
                id_2 = self.tokenizer_2.convert_tokens_to_ids(modifier_tokens[0])
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))
                self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))
                token_embeds = self.text_encoder.get_input_embeddings().weight.data
                token_embeds[id_1] = st['modifier_token'][modifier_tokens_st[0]][0]    
                token_embeds = self.text_encoder_2.get_input_embeddings().weight.data
                token_embeds[id_2] = st['modifier_token'][modifier_tokens_st[0]][1]

            for name, params in self.unet.named_parameters():
                if 'attn2' in name:
                    if compress and ('to_k' in name or 'to_v' in name):
                        params.data += st['unet'][name]['u']@st['unet'][name]['v']
                    elif name in st['unet']:
                        if not deep_replace:
                            params.data.copy_(st['unet'][f'{name}'])
                    else:
                        if deep_replace and ('to_k1' in name or 'to_v1' in name):
                            name_ori = name.replace('to_k1', 'to_k')
                            name_ori = name_ori.replace('to_v1', 'to_v')
                            st['unet'][f'{name}'] = st['unet'][f'{name_ori}']
                            del st['unet'][f'{name_ori}']
                            params.data.copy_(st['unet'][f'{name}']) 
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        concept_list: dict = None,
        stage: Optional[str] = "training",
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        # 0. Default height and width to unet prompt_2:None
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        
        if concept_list['get_concept_bank']:
            if "concept1" in concept_list and stage == "inference":
                prompt1 = [concept_list['concept1']["prompt"]]*concept_list['batch_size']
                (
                    prompt_embeds1,
                    _, _, _,
                ) = self.encode_prompt(
                    prompt=prompt1,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    lora_scale=text_encoder_lora_scale,
                )
            
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        
        if concept_list['get_concept_bank']:
            if "concept1" in concept_list and stage == "inference":
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds1], dim=0)

        prompt_embeds = prompt_embeds.to(device)   # torch.Size([9, 77, 2048])
        add_text_embeds = add_text_embeds.to(device)  #  add_text_embeds.shape torch.Size([6, 1280])
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        refine_info = {}

        concept_bank = {}
        concept_number = concept_list['concept_num']
        main_prompt = concept_list['prompt']
        text_inputs_ids = list(self.tokenizer(
            main_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0])
        text_inputs_ids = text_inputs_ids[1:text_inputs_ids.index(torch.tensor(49407))]
        text_inputs_ids = [int(x) for x in text_inputs_ids]

        noun_list = []
        align_dict = {}
        bind_dict = {}
        for concept_index in range(1, concept_number+1):
            key_index = f"concept{concept_index}"
            class_noun = concept_list[key_index]['class_noun']

            class_noun_ids = list(self.tokenizer(
                class_noun,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids[0])
            class_noun_ids = class_noun_ids[1:class_noun_ids.index(torch.tensor(49407))]
            class_noun_ids = [int(x) for x in class_noun_ids]

            start, end = find_sublist_indices(text_inputs_ids, class_noun_ids)
            concept_list[key_index]['start'], concept_list[key_index]['end'] = start, end+2
            noun_list.append([i for i in range(start+1, end+2)])

            for token_index in range(start, end+2):
                align_dict[str(token_index)] = concept_index - 1
            
            bind_dict[str(start)] = end+1

            print(key_index, concept_list[key_index]['start'], concept_list[key_index]['end'])

            if not concept_list['get_concept_bank']:
                concept_bank[key_index] = torch.load(
                        concept_list[key_index]["concept_bank"],
                        map_location='cpu'
                    )

        concept_list['noun_list'] = noun_list
        concept_list['align_dict'] = align_dict
        concept_list['bind_dict'] = bind_dict

        def hook_fn(module, input, output):
            positive_loss.append(module.processor.positive_loss)
            negative_loss.append(module.processor.negative_loss)

        # register hook
        positive_loss = []
        negative_loss = []

        def register_hook(model):
            for name, layer in model.named_children():
                if isinstance(layer, Attention) and 'attn2' in name:
                    layer.register_forward_hook(hook_fn)
                else:
                    register_hook(layer)
        
        register_hook(self.unet)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # i: 0,1,2,3...
                # t: 1000 ~ 0
                # register_attention_control
                save_attn = False
                refine_attn = False
                attn_procs={}
                attn_dict = {}
                attn_dict['self_attn'] = {}
                attn_dict['cross_attn'] = {}
 
                for name in self.unet.attn_processors.keys():
                    if concept_list['self_rectification']:
                        if i==int((1-concept_list['rectification_start'])*len(timesteps)):
                            save_attn = True
                        elif i > int((1-concept_list['rectification_start'])*len(timesteps)) and len(refine_info)!=0:
                            refine_attn = True

                    attn_procs[name] = MultiConceptprocessor(name=name, step=t, save_attn=save_attn, refine_attn=refine_attn, refine_info=refine_info, attn_dict=attn_dict, concept_list=concept_list, concept_bank=concept_bank, inf_batch_size=prompt_embeds.shape[0])  ###
                
                self.unet.set_attn_processor(attn_procs)

                # blending_guidance
                if concept_list['blending_guidance']:
                    guidance_steps = concept_list['guidance_steps']
                    guidance_stepsize = concept_list['guidance_stepsize']
                    #step_size = torch.pow(t/1000,5) * 10
                    if i < guidance_steps:
                        with torch.enable_grad():
                            latents_grad = latents.clone().detach().requires_grad_(True)
                            #latents_grad = self.scheduler.scale_model_input(latents_grad, t)
                            added_cond_kwargs = {"text_embeds": add_text_embeds[prompt_embeds.shape[0]//2:], "time_ids": add_time_ids[prompt_embeds.shape[0]//2:]}
                            
                            self.unet(
                                latents_grad,
                                t,
                                encoder_hidden_states=prompt_embeds[prompt_embeds.shape[0]//2:],
                                cross_attention_kwargs=cross_attention_kwargs,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0]

                            self.unet.zero_grad()
                            loss = sum(positive_loss) + sum(negative_loss)

                            grad_cond = torch.autograd.grad(
                                    loss.requires_grad_(True), [latents_grad], retain_graph=False, create_graph=False
                                )[0]
                            latents = latents - guidance_stepsize * grad_cond.detach()

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                torch.cuda.empty_cache()
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if concept_list['get_concept_bank']:
                    torch.save(save_weight, concept_list['concept_path'])

                # calculate layout
                if save_attn:
                        creg_masks_all = []
                        creg_objmasks_all = []
                        for i in range(int((noise_pred.shape[0])/2)):
                            merged_dict_cross = {}
                            merged_dict_self = {}
                            for key in attn_dict['cross_attn'].keys():
                                merged_dict_cross[key] = attn_dict['cross_attn'][key][i].unsqueeze(0)
                            
                            for key in attn_dict['self_attn'].keys():
                                merged_dict_self[key] = attn_dict['self_attn'][key][i].unsqueeze(0)
                            
                            token_list = concept_list['noun_list'] # [torch.tensor([2]), torch.tensor([6,7])]
                            token_list = [torch.tensor(i) for i in token_list]

                            layout=get_token_maps(merged_dict_self, merged_dict_cross,"./",
                                                    64, 64, token_list, 2,
                                                    segment_threshold=0.4, num_segments=8, index=i)
                            del merged_dict_self
                            del merged_dict_cross

                            align_dict = concept_list['align_dict']

                            full_layout = 1-layout[-1]

                            mask_64 = []
                            mask_object = []
                            for t_index in range(77):
                                t_index = str(t_index)
                                if t_index in align_dict:
                                    mask_64.append(layout[align_dict[t_index]].squeeze().unsqueeze(0))
                                    mask_object.append((full_layout - layout[align_dict[t_index]]).squeeze().unsqueeze(0))
                                else:
                                    mask_64.append(layout[-1].squeeze().unsqueeze(0)*0)
                                    mask_object.append(layout[-1].squeeze().unsqueeze(0)*0)
                            mask_64 = torch.cat(mask_64,dim=0).unsqueeze(0) # mask_64: [1, 77, 64, 64]
                            mask_object = torch.cat(mask_object,dim=0).unsqueeze(0)

                            print("mask 64",mask_64.shape)
                            creg_mask = {}
                            creg_mask_obj = {}
                            for r in range(2):
                                res=int(64/np.power(2,r))
                                layout_c = F.interpolate(mask_64,(res,res),mode="nearest").view(1, 77, -1)    ###### wrong .view(1, -1, 77)
                                layout_c = rearrange(layout_c, "a b c -> a c b")
                                creg_mask[np.power(res,2)]=layout_c

                                layout_obj = F.interpolate(mask_object,(res,res),mode="nearest").view(1, 77, -1)
                                layout_obj = rearrange(layout_obj, "a b c -> a c b")
                                creg_mask_obj[np.power(res,2)]=layout_obj

                            creg_masks_all.append(creg_mask)
                            creg_objmasks_all.append(creg_mask_obj)
                        
                        creg_masks_dict = {}
                        creg_objmasks_dict = {}
                        for key in creg_masks_all[0].keys():
                            tmp_list1 = []
                            tmp_list2 = []
                            for dict_index in range(len(creg_masks_all)):
                                tmp_list1.append(creg_masks_all[dict_index][key].unsqueeze(0))
                                tmp_list2.append(creg_objmasks_all[dict_index][key].unsqueeze(0))

                            creg_masks_dict[key] = rearrange(torch.cat(tmp_list1, dim=0), "a b c d -> (a b) c d")
                            creg_objmasks_dict[key] = rearrange(torch.cat(tmp_list2, dim=0), "a b c d -> (a b) c d")

                        for key in creg_masks_dict.keys():
                            creg_masks_dict[key] = creg_masks_dict[key].to(device)
                            creg_objmasks_dict[key] = creg_objmasks_dict[key].to(device)

                        refine_info['creg_mask'] = creg_masks_dict 
                        refine_info['creg_mask_obj'] = creg_objmasks_dict

                """ token_mask = torch.zeros(77)
                align_dict = concept_list['align_dict'] #####

                #for key in align_dict:
                #    token_mask[int(key)] = 1
                #token_mask[0] = 1
                for inj in range(1,15):
                    token_mask[inj] = 1
                refine_info['token_mask'] = token_mask
                #torch.save(creg_masks_dict, 'creg_masks_dict.pt') """

                # perform classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    noise_pred_mem = noise_pred_text - noise_pred_uncond

                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

def get_token_maps(selfattn_maps, crossattn_maps, save_dir, width, height, obj_tokens, seed=0, tokens_vis=None,
                   preprocess=False, segment_threshold=0.3, num_segments=5, return_vis=False, save_attn=False, index = 0):
    r"""Function to visualize attention maps.
    Args:
        save_dir (str): Path to save attention maps
        batch_size (int): Batch size
        sampler_order (int): Sampler order
    """

    # create the segmentation mask using self-attention maps
    resolution = 64
    attn_maps_1024 = {8: [], 16: [], 32: [], 64: []}
    for attn_map in selfattn_maps.values():
        resolution_map = np.sqrt(attn_map.shape[1]).astype(int)
        attn_map = attn_map.mean(dim=0)

        if resolution_map != resolution:
            continue
        attn_map = attn_map.reshape(
            1, resolution_map, resolution_map, resolution_map**2).permute([3, 0, 1, 2]).float()
        attn_map = torch.nn.functional.interpolate(attn_map, (resolution, resolution),
                                                mode='bicubic', antialias=True)
        attn_maps_1024[resolution_map].append(attn_map.permute([1, 2, 3, 0]).reshape(
            1, resolution**2, resolution_map**2))
    attn_maps_1024 = torch.cat([torch.cat(v).mean(0).cpu()
                                for v in attn_maps_1024.values() if len(v) > 0], -1).numpy()
    if save_attn:
        print('saving self-attention maps...', attn_maps_1024.shape)
        torch.save(torch.from_numpy(attn_maps_1024),
                   'results/maps/selfattn_maps.pth')
    #seed_everything(seed)rand
 
    sc = SpectralClustering(num_segments, affinity='precomputed', n_init=100,
                            assign_labels='kmeans')
    clusters = sc.fit_predict(attn_maps_1024)
    clusters = clusters.reshape(resolution, resolution)
 
    fig = plt.figure()
    plt.imshow(clusters)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'segmentation_k%d_seed%d_index%d.jpg' % (num_segments, seed, index)),
                bbox_inches='tight', pad_inches=0)
    if return_vis:
        canvas = fig.canvas
        canvas.draw()
        cav_width, cav_height = canvas.get_width_height()
        segments_vis = np.frombuffer(canvas.tostring_rgb(),
                                     dtype='uint8').reshape((cav_height, cav_width, 3))

    plt.close()
    # label the segmentation mask using cross-attention maps
    cross_attn_maps_1024 = []
    for attn_map in crossattn_maps.values():
        resolution_map = np.sqrt(attn_map.shape[1]).astype(int)
        attn_map = attn_map.mean(dim=0)
        attn_map = attn_map.reshape(
            1, resolution_map, resolution_map, -1).permute([0, 3, 1, 2]).float()
        attn_map = torch.nn.functional.interpolate(attn_map, (resolution, resolution),
                                                   mode='bicubic', antialias=True)
        cross_attn_maps_1024.append(attn_map.permute([0, 2, 3, 1]))

    cross_attn_maps_1024 = torch.cat(
        cross_attn_maps_1024).mean(0).cpu().numpy()
    if save_attn:
        print('saving cross-attention maps...', cross_attn_maps_1024.shape)
        torch.save(torch.from_numpy(cross_attn_maps_1024),
                   'results/maps/crossattn_maps.pth')
    normalized_span_maps = []
    for token_ids in obj_tokens:
        span_token_maps = cross_attn_maps_1024[:, :, token_ids.numpy()]
        normalized_span_map = np.zeros_like(span_token_maps)
        for i in range(span_token_maps.shape[-1]):
            curr_noun_map = span_token_maps[:, :, i]
            normalized_span_map[:, :, i] = (
                curr_noun_map - np.abs(curr_noun_map.min())) / (curr_noun_map.max()-curr_noun_map.min())
        normalized_span_maps.append(normalized_span_map)
    foreground_token_maps = [np.zeros([clusters.shape[0], clusters.shape[1]]).squeeze(
    ) for normalized_span_map in normalized_span_maps]

    background_map = np.zeros([clusters.shape[0], clusters.shape[1]]).squeeze()
    match_dict = {} #key: token_index, value: seg_num
    cluster_dict = {}
    max_dict = {}
    data_dict = {}
    for c in range(num_segments):

        cluster_mask = np.zeros_like(clusters)
        cluster_mask[clusters == c] = 1.
        cluster_dict[c] = cluster_mask
        #is_foreground = False
        token_index = 0
        for normalized_span_map, foreground_nouns_map, token_ids in zip(normalized_span_maps, foreground_token_maps, obj_tokens):
            score_maps = [cluster_mask * normalized_span_map[:, :, i]
                          for i in range(len(token_ids))]
            scores = [score_map.sum() / cluster_mask.sum()
                      for score_map in score_maps]
            
            key_now = (c, token_index)
            data_dict[key_now] = scores

            """ if max(scores) > segment_threshold:
                if match_dict[c] == -1:
                    match_dict[c] = token_index
                    max_dict[c] = max(scores)
                else:
                    if max(scores) > max_dict[c]:
                        match_dict[c] = token_index
                        max_dict[c] = max(scores) """

            token_index += 1
            #foreground_nouns_map += cluster_mask
            #is_foreground = True
        
        """ if match_dict[c] == -1:
            background_map += cluster_mask
        else:
            foreground_token_maps[match_dict[c]] += cluster_mask """
        #if not is_foreground:
            #background_map += cluster_mask 
    
    sorted_data = OrderedDict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True))
    #print("sorted", sorted_data)
    for key, value in sorted_data.items():
        if len(match_dict) == len(obj_tokens):
            break
        seg_num = key[0]
        token_index = key[1]
        if seg_num in match_dict.values() or token_index in match_dict.keys():
            continue
        match_dict[token_index] = seg_num
        #print("matched", match_dict)
        #foreground_token_maps[token_index] += cluster_dict[seg_num]
        for token_key in range(len(foreground_token_maps)):
            if token_key == token_index:
                continue
            foreground_token_maps[token_key] += cluster_dict[seg_num]
        background_map += cluster_dict[seg_num]

    background_map = 1-background_map
    """
    for c in range(num_segments):
        cluster_mask = np.zeros_like(clusters)
        cluster_mask[clusters == c] = 1.
        is_foreground = False
        for normalized_span_map, foreground_nouns_map, token_ids in zip(normalized_span_maps, foreground_token_maps, obj_tokens):
            score_maps = [cluster_mask * normalized_span_map[:, :, i]
                          for i in range(len(token_ids))]
            scores = [score_map.sum() / cluster_mask.sum()
                      for score_map in score_maps]
            if max(scores) > segment_threshold:
                foreground_nouns_map += cluster_mask
                is_foreground = True
        if not is_foreground:
            background_map += cluster_mask"""

    foreground_token_maps.append(background_map)

    # resize the token maps and visualization
    resized_token_maps = torch.cat([torch.nn.functional.interpolate(torch.from_numpy(token_map).unsqueeze(0).unsqueeze(
        0), (height, width), mode='bicubic', antialias=True)[0] for token_map in foreground_token_maps]).clamp(0, 1)

    resized_token_maps = resized_token_maps / \
        (resized_token_maps.sum(0, True)+1e-8)
    resized_token_maps = [token_map.unsqueeze(
        0) for token_map in resized_token_maps]
    foreground_token_maps = [token_map[None, :, :]
                             for token_map in foreground_token_maps]
    token_maps_vis = plot_attention_maps([foreground_token_maps, resized_token_maps], obj_tokens,
                                         save_dir, seed, tokens_vis, index)
    resized_token_maps = [token_map.unsqueeze(1).repeat(
        [1, 1, 1, 1]).to(attn_map.dtype).cuda() for token_map in resized_token_maps]
    if return_vis:
        return resized_token_maps, segments_vis, token_maps_vis
    else:
        return resized_token_maps

def plot_attention_maps(atten_map_list, obj_tokens, save_dir, seed, tokens_vis=None, index=None):
    atten_names = ['presoftmax', 'postsoftmax', 'postsoftmax_erosion']
    for i, attn_map in enumerate(atten_map_list):
        n_obj = len(attn_map)
        plt.figure()
        plt.clf()

        fig, axs = plt.subplots(
            ncols=n_obj+1, gridspec_kw=dict(width_ratios=[1 for _ in range(n_obj)]+[0.1]))

        fig.set_figheight(3)
        fig.set_figwidth(3*n_obj+0.1)

        cmap = plt.get_cmap('OrRd')

        vmax = 0
        vmin = 1
        for tid in range(n_obj):
            attention_map_cur = attn_map[tid]
            vmax = max(vmax, float(attention_map_cur.max()))
            vmin = min(vmin, float(attention_map_cur.min()))

        for tid in range(n_obj):
            sns.heatmap(
                attn_map[tid][0], annot=False, cbar=False, ax=axs[tid],
                cmap=cmap, vmin=vmin, vmax=vmax
            )
            axs[tid].set_axis_off()

            if tokens_vis is not None:
                if tid == n_obj-1:
                    axs_xlabel = 'other tokens'
                else:
                    axs_xlabel = ''
                    for token_id in obj_tokens[tid]:
                        axs_xlabel += ' ' + tokens_vis[token_id.item() -
                                                       1][:-len('</w>')]
                axs[tid].set_title(axs_xlabel)

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, cax=axs[-1])
        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        img = np.frombuffer(canvas.tostring_rgb(),
                            dtype='uint8').reshape((height, width, 3))

        fig.tight_layout()

        plt.savefig(os.path.join(
            save_dir, 'average_index%d_attn%d.png' % (index, i)), dpi=100)
        plt.close('all')
    return img