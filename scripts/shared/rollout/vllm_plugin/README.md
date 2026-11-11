# vLLM plugin metadata

`aadistill_vllm_plugin-1.0.dist-info/` declares one vLLM general plugin,
`aadistill_projected_residual`, whose entry point
`aadistill.initialization.adapters.projected_residual_vllm:register` registers the
projected-residual family (its Hugging Face config class and its vLLM model
class) in every vLLM process.

Nothing is installed. A vLLM interpreter discovers the entry point when this
directory and the repository's `src` are on its `PYTHONPATH`
(`shared.rollout.vllm_plugin_path.vllm_plugin_pythonpath`); `importlib.metadata`
reads distributions from every `sys.path` entry. Set `VLLM_PLUGINS` to
restrict which plugins load.
