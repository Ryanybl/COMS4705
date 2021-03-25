import numpy as np

import torch
from transformers import BertTokenizer, BertModel


def initialize_models():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()

    return model, tokenizer


def get_embeddings_from_text(text, tokenizer, model):
    """
        Wrapper around token extraction and embedding generation, converted to ndarray
    :param text:
    :param tokenizer:
    :param model:
    :return:
    """
    tokenized_text, token_tensor, segment_tensor = get_bert_inputs(text, tokenizer)
    return get_bert_embeddings(token_tensor, segment_tensor, model).squeeze().numpy()


def get_cls_embeddings_from_text(text, tokenizer, model):
    """
        Wrapper around token extraction and embedding generation, converted to ndarray for the CLS token
    :param text:
    :param tokenizer:
    :param model:
    :return:
    """
    tokenized_text, token_tensor, segment_tensor = get_bert_inputs(text, tokenizer)
    return get_bert_cls_embeddings(token_tensor, segment_tensor, model).squeeze().numpy()


"""

"""
def get_bert_inputs(text, tokenizer):
    capped_text = text#"[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(capped_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # I think this is used to distinguish between multiple sentences
    segment_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segment_ids])

    return tokenized_text, tokens_tensor, segments_tensor


def get_bert_embeddings(tokens_tensor, segments_tensor, model):

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensor)
        hidden = outputs[2][1:]

    #embeddings = hidden[-1]
    embeddings = outputs[0]
    #print(embeddings.shape)
    avg_emb = torch.mean(embeddings, dim=1)
    #print(avg_emb.shape)
    #embeddings = torch.squeeze(embeddings, dim=0)
    #print(embeddings.shape)
    return avg_emb


def get_bert_cls_inputs(text, tokenizer):
    capped_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(capped_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # I think this is used to distinguish between multiple sentences
    segment_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segment_ids])

    return tokenized_text, tokens_tensor, segments_tensor


def get_bert_cls_embeddings(tokens_tensor, segments_tensor, model):

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensor)
        hidden = outputs[2][1:]
        all_embeddings = torch.stack(hidden, dim=0)
        all_embeddings = torch.squeeze(all_embeddings, dim=1)
        all_embeddings = all_embeddings.permute(1, 0, 2)

    summed = torch.sum(all_embeddings[0][-4:], dim=0)
    #print(summed.shape)
    embeddings = outputs[0].squeeze()[0]
    #print(embeddings.shape)
    #embeddings = torch.squeeze(embeddings, dim=0)
    #print(embeddings.shape)
    return summed


def test_run():
    model, tokenizer = initialize_models()

    # this example wasn't working, but it turned out to be a tokenization bug
    tokenized_text1, tokens_tensor1, segments_tensor1 = get_bert_inputs("professor", tokenizer)
    print(tokenized_text1)
    tokenized_text2, tokens_tensor2, segments_tensor2 = get_bert_inputs("cucumber", tokenizer)
    print(tokenized_text2)

    v1 = get_bert_embeddings(tokens_tensor1, segments_tensor1, model).squeeze().numpy()
    v2 = get_bert_embeddings(tokens_tensor2, segments_tensor2, model).squeeze().numpy()
    print(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    #print(embeddings)
    #print(avg_emb.shape)

#test_run()
