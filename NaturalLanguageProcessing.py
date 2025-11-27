# Use a tokenizer to prepare text for a neural network
# See how embeddings are used to identify numerical features for text data

# Tokenization
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

text_1 = "I understand equations, both the simple and quadratical."
text_2 = "What kind of equations do I understand?"

# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)
indexed_tokens

# We can use convert_ids_to_tokens to see what was used as tokens.
tokenizer.convert_ids_to_tokens([str(token) for token in indexed_tokens])

# If we want to decode our encoded text directly
tokenizer.decode(indexed_tokens)

#  let's define which index correspnds to which special token.
cls_token = 101
sep_token = 102


def get_segment_ids(indexed_tokens):
    segment_ids = []
    segment_id = 0
    for token in indexed_tokens:
        if token == sep_token:
            segment_id += 1
        segment_ids.append(segment_id)
    segment_ids[-1] -= 1  # Last [SEP] is ignored
    return torch.tensor([segment_ids]), torch.tensor([indexed_tokens])

# Does each number correctly correspond to the first and second sentence?
segments_tensors, tokens_tensor = get_segment_ids(indexed_tokens)
segments_tensors

# Text Masking
tokenizer.mask_token
tokenizer.mask_token_id

masked_index = 5

# we'll apply the mask and verify it appears in our sequence of setences.
indexed_tokens[masked_index] = tokenizer.mask_token_id
tokens_tensor = torch.tensor([indexed_tokens])
tokenizer.decode(indexed_tokens)

# Then, we will load the model used to predict the missing word:
masked_lm_model = BertForMaskedLM.from_pretrained("bert-base-cased")
masked_lm_model

embedding_table = next(masked_lm_model.bert.embeddings.word_embeddings.parameters())
embedding_table

# to verify there is an embedding of size 768 for each of the 28996 tokens in BERT's vocabulary.
embedding_table.shape

# testing the model with no grad
with torch.no_grad():
    predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)
predictions

predictions[0].shape

# Get the predicted token
predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
predicted_index

predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
predicted_token

tokenizer.decode(indexed_tokens)

# Question and Answering
text_1 = "I understand equations, both the simple and quadratical."
text_2 = "What kind of equations do I understand?"

question_answering_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
segments_tensors, tokens_tensor = get_segment_ids(indexed_tokens)

question_answering_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Predict the start and end positions logits
with torch.no_grad():
    out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)
out

out.start_logits
out.end_logits

answer_sequence = indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1]
answer_sequence

question_answering_tokenizer.convert_ids_to_tokens(answer_sequence)

question_answering_tokenizer.decode(answer_sequence)















