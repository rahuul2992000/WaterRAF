#chronos_bolt_adapter.py

import torch
from chronos import BaseChronosPipeline


class ChronosBoltAdapter:
    """
    Adapter so existing CODE 2 (which expects a sample-path interface)
    can use Chronos-Bolt (quantile model) seamlessly.

    - predict(...) returns synthetic samples (B, num_samples, H) by
      tiling the *mean* across Bolt's quantile forecasts (mirrors CODE 1,
      where you used the 'mean' column).
    - embed(...) returns simple z-scored embeddings (B, T, 1) plus (means, stds)
      so your retrieval code still works. (You can later replace with a true
      model embedding if exposed.)
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-small",
        device_map: str = "cuda",
        torch_dtype=None,
        faux_num_samples: int = 20,
    ):
        if torch_dtype is None:
            torch_dtype = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
        self.inner = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.faux_num_samples = faux_num_samples
        self.quantiles = getattr(self.inner, "quantiles", None)

        # Default context length if not discoverable
        self.context_length = 2048
        model = getattr(self.inner, "model", None)
        if model and hasattr(model, "config"):
            cc = getattr(model.config, "chronos_config", None)
            if isinstance(cc, dict):
                self.context_length = cc.get("context_length", self.context_length)
            elif cc is not None:
                self.context_length = getattr(
                    cc, "context_length", self.context_length
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_batch(self, context):
        """
        Accepts:
          - 1D tensor
          - list of 1D tensors (variable lengths)
          - 2D tensor (already padded)
        Produces left-padded (NaN) 2D tensor (B, T).
        """
        if isinstance(context, list):
            max_len = max(len(c) for c in context)
            batch = []
            for c in context:
                if not isinstance(c, torch.Tensor) or c.ndim != 1:
                    raise ValueError("Each list element must be a 1D torch.Tensor.")
                if len(c) < max_len:
                    pad = torch.full(
                        (max_len - len(c),),
                        float("nan"),
                        dtype=c.dtype,
                        device=c.device,
                    )
                    c = torch.cat([pad, c], dim=0)
                batch.append(c)
            ctx = torch.stack(batch)
        elif isinstance(context, torch.Tensor):
            if context.ndim == 1:
                ctx = context.unsqueeze(0)
            elif context.ndim == 2:
                ctx = context
            else:
                raise ValueError(f"Unsupported context tensor shape {context.shape}")
        else:
            raise ValueError("Context must be list[tensor] or tensor.")

        # Truncate from left if exceeding model context length
        if ctx.shape[-1] > self.context_length:
            ctx = ctx[:, -self.context_length:]
        return ctx

    # ------------------------------------------------------------------
    # Public API expected by CODE 2
    # ------------------------------------------------------------------
    def embed(self, context):
        """
        Returns (embeddings, (means, stds)).
        Embeddings: simple z-scored values -> (B, T, 1).
        """
        ctx = self._prepare_batch(context)
        mask = ~torch.isnan(ctx)
        filled = torch.nan_to_num(ctx)

        mean = (filled * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(
            min=1
        )
        var = (
            ((filled - mean) ** 2) * mask
        ).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
        std = var.sqrt().clamp(min=1e-6)

        z = (filled - mean) / std
        emb = z.unsqueeze(-1)  # (B, T, 1)

        return emb, (mean.squeeze(-1).cpu(), std.squeeze(-1).cpu())

    def predict(
        self,
        context,
        prediction_length: int,
        num_samples: int = None,
        **_ignored,
    ):
        """
        Returns synthetic samples shaped (B, num_samples, H) where each "sample"
        is the same mean point forecast (mirrors CODE 1's use of 'mean').

        - Bolt underlying predict => (B, Q, H) or (Q, H)
        - We average across the quantile dimension to get the mean.
        """
        if num_samples is None:
            num_samples = self.faux_num_samples

        ctx = self._prepare_batch(context)
        q_forecast = self.inner.predict(ctx, prediction_length=prediction_length)

        # Normalize shape
        if q_forecast.dim() == 2:  # (Q, H)
            q_forecast = q_forecast.unsqueeze(0)  # (1, Q, H)

        if q_forecast.dim() != 3:
            raise RuntimeError(f"Unexpected Bolt forecast shape {q_forecast.shape}")

        # Mean across quantiles (uniform; evenly spaced quantiles)
        point = q_forecast.mean(dim=1)  # (B, H)

        samples = point.unsqueeze(1).repeat(1, num_samples, 1)
        return samples
