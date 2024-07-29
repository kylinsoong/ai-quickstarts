import tensorflow as tf

vocab = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
print(f"vocabulary: {len(vocab)} unique characters")

texts = ["abcdefg", "xyz", "hello", "world", "KYLIN", "SOONG"]
print("original text:", texts)

chars = tf.strings.unicode_split(texts, input_encoding="UTF-8")
print("text to token:",chars)

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)
ids = ids_from_chars(chars)
print("token to ids:", ids)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)
chars = chars_from_ids(ids)
print("ids to token:", chars)

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

print("text from ids:", text_from_ids(ids))
