defmodule Diffusers.Layers do
  # class GEGLU(keras.layers.Layer):
  #     def __init__(self, dim_out):
  #         super().__init__()
  #         self.proj = keras.layers.Dense(dim_out * 2)
  #         self.dim_out = dim_out

  #     def call(self, x):
  #         xp = self.proj(x)
  #         x, gate = xp[..., : self.dim_out], xp[..., self.dim_out :]
  #         return x * gelu(gate)
  def geglu(x, units, opts \\ []) do
    xp = Axon.dense(x, units * 2)
    x = Axon.nx(xp, &Nx.slice(&1, [0], [units]))
    gate = Axon.nx(xp, &Nx.slice(&1, [units], []))
    x * Diffusers.Activations.gelu(gate)
  end

  # class Downsample(keras.layers.Layer):
  #     def __init__(self, channels):
  #         super().__init__()
  #         self.op = PaddedConv2D(channels, 3, stride=2, padding=1)

  #     def call(self, x):
  #         return self.op(x)
  def downsample(x, units, opts \\ []) do
    Axon.conv(x, units, 3, strides: 2, padding: 1)
  end

  # class ResBlock(keras.layers.Layer):
  #     def __init__(self, channels, out_channels):
  #         super().__init__()
  #         self.in_layers = [
  #             tfa.layers.GroupNormalization(epsilon=1e-5),
  #             keras.activations.swish,
  #             PaddedConv2D(out_channels, 3, padding=1),
  #         ]
  #         self.emb_layers = [
  #             keras.activations.swish,
  #             keras.layers.Dense(out_channels),
  #         ]
  #         self.out_layers = [
  #             tfa.layers.GroupNormalization(epsilon=1e-5),
  #             keras.activations.swish,
  #             PaddedConv2D(out_channels, 3, padding=1),
  #         ]
  #         self.skip_connection = (
  #             PaddedConv2D(out_channels, 1) if channels != out_channels else lambda x: x
  #         )

  #     def call(self, inputs):
  #         x, emb = inputs
  #         h = apply_seq(x, self.in_layers)
  #         emb_out = apply_seq(emb, self.emb_layers)
  #         h = h + emb_out[:, None, None]
  #         h = apply_seq(h, self.out_layers)
  #         ret = self.skip_connection(x) + h
  #         return ret
  # end
  def residual(x, embedded, units, _opts \\ []) do
    h =
      x
      |> Axon.group_norm(32, epsilon: 1.0e-5)
      |> Axon.silu()
      |> Axon.conv(units, 3, paddind: 1)

    embedded =
      embedded
      |> Axon.silu()
      |> Axon.dense(units)

    (h + embedded)
    |> Axon.group_norm(32, epsilon: 1.0e-5)
    |> Axon.silu()
    |> Axon.conv(units, 3, padding: 1)
  end

  # class SpatialTransformer(keras.layers.Layer):
  #     def __init__(self, channels, n_heads, d_head):
  #         super().__init__()
  #         self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
  #         assert channels == n_heads * d_head
  #         self.proj_in = PaddedConv2D(n_heads * d_head, 1)
  #         self.transformer_blocks = [BasicTransformerBlock(channels, n_heads, d_head)]
  #         self.proj_out = PaddedConv2D(channels, 1)

  #     def call(self, inputs):
  #         x, context = inputs
  #         b, h, w, c = x.shape
  #         x_in = x
  #         x = self.norm(x)
  #         x = self.proj_in(x)
  #         x = tf.reshape(x, (-1, h * w, c))
  #         for block in self.transformer_blocks:
  #             x = block([x, context])
  #         x = tf.reshape(x, (-1, h, w, c))
  #         return self.proj_out(x) + x_in
  def spatial_transformer(x, units, heads, head_size, opts \\ []) do
    {_, h, w, c} = Nx.shape(x)

    x =
      x
      |> Axon.group_norm(32, epsilon: 1.0e-5)
      |> Axon.conv(units, 1)
      |> Axon.reshape({-1, h * w, c})
      |> Axon.transformer(units, heads, head_size)
      |> Axon.reshape({-1, h, w, c})
      |> Axon.conv(units, 1)
  end

  # class BasicTransformerBlock(keras.layers.Layer):
  #     def __init__(self, dim, n_heads, d_head):
  #         super().__init__()
  #         self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
  #         self.attn1 = CrossAttention(n_heads, d_head)

  #         self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
  #         self.attn2 = CrossAttention(n_heads, d_head)

  #         self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
  #         self.geglu = GEGLU(dim * 4)
  #         self.dense = keras.layers.Dense(dim)

  #     def call(self, inputs):
  #         x, context = inputs
  #         x = self.attn1([self.norm1(x)]) + x
  #         x = self.attn2([self.norm2(x), context]) + x
  #         return self.dense(self.geglu(self.norm3(x))) + x
  def transformer(x, units, heads, head_size, opts \\ []) do
    x =
      x
      |> Axon.layer_norm(epsilon: 1.0e-5)
      |> Axon.cross_attention(heads, head_size)
      |> Axon.add(x)

    x =
      x
      |> Axon.layer_norm(epsilon: 1.0e-5)
      |> Axon.cross_attention(heads, head_size)
      |> Axon.add(x)

    x
    |> Axon.layer_norm(epsilon: 1.0e-5)
    |> geglu(units * 4)
    |> Axon.dense(units)
    |> Axon.add(x)
  end

  # class CrossAttention(keras.layers.Layer):
  #     def __init__(self, n_heads, d_head):
  #         super().__init__()
  #         self.to_q = keras.layers.Dense(n_heads * d_head, use_bias=False)
  #         self.to_k = keras.layers.Dense(n_heads * d_head, use_bias=False)
  #         self.to_v = keras.layers.Dense(n_heads * d_head, use_bias=False)
  #         self.scale = d_head**-0.5
  #         self.num_heads = n_heads
  #         self.head_size = d_head
  #         self.to_out = [keras.layers.Dense(n_heads * d_head)]

  #     def call(self, inputs):
  #         assert type(inputs) is list
  #         if len(inputs) == 1:
  #             inputs = inputs + [None]
  #         x, context = inputs
  #         context = x if context is None else context
  #         q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
  #         assert len(x.shape) == 3
  #         q = tf.reshape(q, (-1, x.shape[1], self.num_heads, self.head_size))
  #         k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
  #         v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

  #         q = keras.layers.Permute((2, 1, 3))(q)  # (bs, num_heads, time, head_size)
  #         k = keras.layers.Permute((2, 3, 1))(k)  # (bs, num_heads, head_size, time)
  #         v = keras.layers.Permute((2, 1, 3))(v)  # (bs, num_heads, time, head_size)

  #         score = td_dot(q, k) * self.scale
  #         weights = keras.activations.softmax(score)  # (bs, num_heads, time, time)
  #         attention = td_dot(weights, v)
  #         attention = keras.layers.Permute((2, 1, 3))(
  #             attention
  #         )  # (bs, time, num_heads, head_size)
  #         h_ = tf.reshape(attention, (-1, x.shape[1], self.num_heads * self.head_size))
  #         return apply_seq(h_, self.to_out)
  def cross_attention(x, context, heads, head_size, opts \\ []) do
    q =
      x
      |> Axon.dense(heads * head_size, use_bias: false)
      |> Axon.reshape({-1, elem(Nx.shape(x), 1), heads, head_size})
      |> Axon.transpose({2, 1, 3})

    k =
      context
      |> Axon.dense(heads * head_size, use_bias: false)
      |> Axon.reshape({-1, elem(Nx.shape(context), 1), heads, head_size})
      |> Axon.transpose({2, 3, 1})

    v =
      context
      |> Axon.dense(heads * head_size, use_bias: false)
      |> Axon.reshape({-1, elem(Nx.shape(context), 1), heads, head_size})
      |> Axon.transpose({2, 1, 3})

    score = Nx.dot(q, k) * head_size ** -0.5
    weights = Axon.softmax(score)

    Nx.dot(weights, v)
    |> Axon.transpose({2, 1, 3})
    |> Axon.reshape({-1, Nx.shape(x)[1], heads * head_size})
    |> Axon.dense(heads * head_size)
  end

  # def td_dot(a, b):
  #     aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
  #     bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
  #     cc = keras.backend.batch_dot(aa, bb)
  #     return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))
  defn td_dot(a, b) do
    a = Nx.reshape(a, {-1, elem(Nx.shape(a), 2), elem(Nx.shape(a), 3)})
    b = Nx.reshape(b, {-1, elem(Nx.shape(b), 2), elem(Nx.shape(b), 3)})

    Nx.dot(a, b)
    |> Nx.reshape({-1, elem(Nx.shape(a), 1), elem(Nx.shape(b), 1), elem(Nx.shape(b), 2)})
  end
end
