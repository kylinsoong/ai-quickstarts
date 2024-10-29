import tensorflow as tf

text = """I

Living is no laughing matter:
	you must live with great seriousness
		like a squirrel, for example—
   I mean without looking for something beyond and above living,
		I mean living must be your whole occupation.
Living is no laughing matter:
	you must take it seriously,
	so much so and to such a degree
   that, for example, your hands tied behind your back,
                                            your back to the wall,
   or else in a laboratory
	in your white coat and safety glasses,
	you can die for people—
   even for people whose faces you’ve never seen,
   even though you know living
	is the most real, the most beautiful thing.
I mean, you must take living so seriously
   that even at seventy, for example, you’ll plant olive trees—
   and not for your children, either,
   but because although you fear death you don’t believe it,
   because living, I mean, weighs heavier.
II

Let’s say we’re seriously ill, need surgery—
which is to say we might not get up
			from the white table.
Even though it’s impossible not to feel sad
			about going a little too soon,
we’ll still laugh at the jokes being told,
we’ll look out the window to see if it’s raining,
or still wait anxiously
		for the latest newscast. . . 
Let’s say we’re at the front—
	for something worth fighting for, say.
There, in the first offensive, on that very day,
	we might fall on our face, dead.
We’ll know this with a curious anger,
        but we’ll still worry ourselves to death
        about the outcome of the war, which could last years.
Let’s say we’re in prison
and close to fifty,
and we have eighteen more years, say,
                        before the iron doors will open.
We’ll still live with the outside,
with its people and animals, struggle and wind—
                                I  mean with the outside beyond the walls.
I mean, however and wherever we are,
        we must live as if we will never die.
III

This earth will grow cold,
a star among stars
               and one of the smallest,
a gilded mote on blue velvet—
	  I mean this, our great earth.
This earth will grow cold one day,
not like a block of ice
or a dead cloud even 
but like an empty walnut it will roll along
	  in pitch-black space . . . 
You must grieve for this right now
—you have to feel this sorrow now—
for the world must be loved this much
                               if you’re going to say “I lived”. . ."""

vocab = sorted(set(text))
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)
all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))
print("length of raw text:", len(text), "length of charater index:", tf.size(all_ids).numpy())
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
seq_length = 50
sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = (
    dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
)

vocab_size = len(ids_from_chars.get_vocabulary())
embedding = tf.keras.layers.Embedding(vocab_size, 256)
gru = tf.keras.layers.GRU(1024, return_sequences=True, return_state=True)
x = embedding(dataset, training=False)
states = gru.get_initial_state(x)
x, states = self.gru(x, initial_state=False, training=False)
