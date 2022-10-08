defmodule Diffusers.Activations do
  import Nx.Defn

  defn gelu(x) do
    0.5 * x * (1 + Axon.Activations.tanh(x * 0.7978845608 * (1 + 0.044715 * x ** 2)))
  end

  defn quick_gelu(x) do
    x * Axon.Activations.sigmoid(x * 1.702)
  end
end
