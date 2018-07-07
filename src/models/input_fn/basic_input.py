import tensorflow as tf


def basic_input_fn(event_texts, target_price, other_prices, final_price, repeat, batch):
    features = {'event_texts': event_texts, 'target_price': target_price, 'other_prices': other_prices}
    labels = final_price
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.repeat(repeat).batch(batch)
