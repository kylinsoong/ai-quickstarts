def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

results = split_input_target(list("Tensorflow"))
print(results)

results = split_input_target(list("hello"))
print(results)

