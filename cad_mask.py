import sys
import random
from typing import Union, List, Optional
from transformers import BertTokenizerFast

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(1002)

class CAD:
    def __init__(self, model_name: str, device: Union[int,str] = 0):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map=device, use_cache=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model_name.startswith('huggyllama'): # add [PAD] token to tokenizer if model_name is huggyllama, because huggyllama doesn't have a pad token
            special_tokens_dict = {'pad_token': '[PAD]'}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
        

    def _top_p_sampling(self, 
                        logits: torch.Tensor, 
                        top_p: float = 0.9, 
                        filter_value: float = -float("Inf"), 
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep - 1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

        return logits


    def _top_k_sampling(self, 
                        logits: torch.Tensor, 
                        top_k: int = 20, 
                        filter_value: float = -float("Inf"), 
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :

        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None] # * logit 값이 Top-k의 토큰 중 가장 작은 값보다 작은 토큰의 인덱스 반환 
        logits[indices_to_remove] = filter_value

        return logits


    def predict_next_token(self, 
                           logits: torch.Tensor, 
                           decoding_strategy: str, 
                           top_p: float, 
                           top_k: int, 
                           use_repetition_penalty: bool, 
                           repetition_penalty_value: float, 
                           generated_tokens: List[set] = None
                           ) -> torch.Tensor :

        # * Repetitin Penalty 참고 코드 : https://huggingface.co/transformers/v2.11.0/_modules/transformers/modeling_utils.html#PreTrainedModel.enforce_repetition_penalty_
        if use_repetition_penalty:
            assert repetition_penalty_value >= 1.0, "Repetition penalty must be >= 1."
            mask = torch.zeros_like(logits)
            for i, token_set in enumerate(generated_tokens):
                mask[i, list(token_set)] = 1.0
            penalty = torch.where(mask == 1.0, repetition_penalty_value, 1.0) # generated_tokens에 있는 토큰들은 penalty를 repetition_penalty_value로, 없는 토큰들은 1.0(현상 유지)으로 설정
            logits *= torch.where(logits < 0, penalty, 1.0/penalty) # if logit is smaller than 0, multiply with penalty, else divide by penalty
        
        if decoding_strategy == 'top_p':
            assert top_p is not None, "top_p must be provided for top_p sampling"
            logits = self._top_p_sampling(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()
            #print(probs)
            return next_token, probs

        elif decoding_strategy == 'top_k':
            assert top_k is not None, "top_k must be provided for top_k sampling"
            logits = self._top_k_sampling(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

        elif decoding_strategy == 'greedy':
            next_token = torch.argmax(logits, dim=-1)

        return next_token

    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p_squeeze = p.squeeze()
        q_squeeze = q.squeeze()
        return torch.sum(p_squeeze * torch.log((p_squeeze + 1e-4) / (q_squeeze + 1e-4)))

    def generate(self, 
                input_texts: List[str], 
                contexts: Optional[List[str]] = None, 
                use_context_aware: bool = True,
                alpha: float = 0.5,
                max_length: int = 256,
                decoding_strategy: str = 'top_p',
                top_p_value: float = 0.9,
                top_k_value: int = 20,
                use_repetition_penalty: bool = False, 
                repetition_penalty_value: float = 1.0,
                num_masks: int = 10  # Number of times to mask the context
                ) -> List[List[int]]:

        # Tokenize 'input_texts' and create attention masks
        tokenized_inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        masked_contexts_nested = []
        input_ids_with_contexts_nested = []
        attention_mask_with_contexts_nested = []

        for i in range(6):
            # Tokenize 'contexts' and mask a random token in each context
            masked_contexts = []
            #print("**************************************")
            MASK_TOKEN = '[MASK]'
            for context in contexts:
                tokens = self.tokenizer.tokenize(context)
                tokens[i] = MASK_TOKEN
                masked_context = self.tokenizer.convert_tokens_to_string(tokens)
                masked_contexts.append(masked_context)
            masked_contexts_nested.append(masked_contexts)
        if contexts and use_context_aware:
            for masked_contexts in masked_contexts_nested:
                inputs_with_contexts = [context + self.tokenizer.eos_token + input_text for context, input_text in zip(masked_contexts, input_texts)]
                #print(inputs_with_contexts)
                tokenized_inputs_with_contexts = self.tokenizer(inputs_with_contexts, return_tensors="pt", padding=True, truncation=True, max_length=256)
                #print(tokenized_inputs_with_contexts)

                input_ids_with_contexts = tokenized_inputs_with_contexts['input_ids']
                input_ids_with_contexts_nested.append(input_ids_with_contexts)
                #print(input_ids_with_contexts)

                attention_mask_with_contexts = tokenized_inputs_with_contexts['attention_mask']
                attention_mask_with_contexts_nested.append(attention_mask_with_contexts)
                #print(attention_mask_with_contexts)
        #else:
            #input_ids_with_contexts = input_ids
            #attention_mask_with_contexts = attention_mask

            cur_len = 0
            batch_size = len(input_ids)
            unfinished_sents = input_ids.new(batch_size).fill_(1)
            sent_lengths = input_ids.new(batch_size).fill_(max_length)

            generated_tokens = [[] for _ in range(batch_size)]
                    # Generate tokens
            with torch.no_grad():
                while cur_len < max_length:
                    #print("*********************************")
                    #print(input_ids_with_contexts)
                    #print(attention_mask_with_contexts)
                    # Generate tokens conditioned on the masked context
                    probs =  []
                    for input_ids_with_contexts, attention_mask_with_contexts in zip(input_ids_with_contexts_nested, attention_mask_with_contexts_nested):
                        outputs = self.model(input_ids_with_contexts, attention_mask=attention_mask)
                        next_token_logits = outputs.logits[:, -1, :]  # Get logits for the next token

                        # Predict next token according to decoding strategy
                        next_token, prob = self.predict_next_token(logits=next_token_logits, 
                                                            decoding_strategy=decoding_strategy, 
                                                            top_p=top_p_value, 
                                                            top_k=top_k_value, 
                                                            use_repetition_penalty=use_repetition_penalty, 
                                                            repetition_penalty_value=repetition_penalty_value, 
                                                            generated_tokens=[set(tokens) for tokens in generated_tokens])

                        # Handle EOS token and padding
                        if self.tokenizer.eos_token_id is not None:
                            tokens_to_add = next_token * unfinished_sents + (self.tokenizer.pad_token_id) * (1 - unfinished_sents)
                        else:
                            tokens_to_add = next_token

                        # Update input_ids and attention masks for the next forward pass
                        input_ids_with_contexts = torch.cat([input_ids_with_contexts, tokens_to_add.unsqueeze(-1)], dim=-1)
                        attention_mask_with_contexts = torch.cat([attention_mask_with_contexts, unfinished_sents.unsqueeze(-1)], dim=-1)

                        cur_len += 1

                        # Update generated tokens and check for completion
                        for i, token in enumerate(tokens_to_add.tolist()):
                            if unfinished_sents[i] == 1:
                                generated_tokens[i].append(token)

                        # Check for sentences that are finished
                        if self.tokenizer.eos_token_id is not None:
                            eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                            unfinished_sents.mul_((~eos_in_sents).long())

                        # Break if all sentences are finished : stop when there is a EOS token in each sentence, or if we exceed the maximum length
                        if unfinished_sents.max() == 0:
                            break
                        probs.append(prob)
            total_kl_div = 0.0
            for i in range(len(probs)):
                for j in range(i + 1, len(probs)):
                    print(torch.sum(probs[i]))
                    print(torch.sum(probs[j]))
                    print(self.kl_divergence(probs[i], probs[j]))
                    total_kl_div += self.kl_divergence(probs[i], probs[j])
            print(total_kl_div)
            print(len(probs))
            mean_kl_div = total_kl_div / len(probs)
            print("Mean KL Divergence:", mean_kl_div)
            exit()
        return generated_tokens, probs, mean_kl_div

cad_model = CAD(model_name="huggyllama/llama-7b", device=4)
contexts = ['Write a quote that ends in the word "early":']
input_texts = ['Better late than']

outputs, probs, mean_kl = cad_model.generate(
                            input_texts=input_texts,
                            use_context_aware=True,
                            contexts=contexts,
                            max_length=20,
                            alpha=0.5,
                            decoding_strategy='top_p',
                            top_p_value=0.9,
                            use_repetition_penalty=True,
                            repetition_penalty_value=1.5,
                            )
#print(outputs)
#print("Probs", probs)
print(len(probs))
exit()
print(cad_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
for i in outputs[0]:
    print(f'Token ID : {i} | Token: {cad_model.tokenizer.decode(i)}')
