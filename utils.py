"""General-purpose plotting utilities; no training/model dependency."""
from collections.abc import Mapping
from functools import wraps
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

__all__ = ["line_plot", "save_image_rows"]


def save_image_rows(rows, output_path):
    """Save comparison rows with unequal image sizes without stretching them."""
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image
    from torchvision.utils import make_grid

    panels = [to_pil_image(make_grid(row.detach().cpu().clamp(0, 1),
                                   nrow=len(row), padding=2, pad_value=1)) for row in rows]
    if not panels:
        raise ValueError('At least one comparison row is required')
    canvas = Image.new('RGB', (max(panel.width for panel in panels),
                               sum(panel.height for panel in panels) + 2 * (len(panels) - 1)), 'white')
    top = 0
    for panel in panels:
        canvas.paste(panel, (0, top))
        top += panel.height + 2
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _values(value: Any, name: str) -> list:
    """Copy a finite column without consuming iterators in the returned result."""
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of values")
    # Supports NumPy and detached CPU/CUDA tensors without importing either.
    if callable(getattr(value, "detach", None)):
        value = value.detach().cpu()
    if callable(getattr(value, "tolist", None)):
        value = value.tolist()
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of values") from exc
    if iterator is value:
        raise TypeError(f"{name} must be reusable data, not a one-shot iterator")
    return list(value)


def _plot_data(data: Any, x: Optional[str], y: Optional[Sequence[str]]):
    """Normalize column mappings, row records or one numeric sequence."""
    if isinstance(data, Mapping):
        columns = data
    else:
        rows = _values(data, "plot data")
        if not rows:
            raise ValueError("plot data must not be empty")
        if isinstance(rows[0], Mapping):
            keys = list(rows[0]) if y is None else list(y)
            if x is not None and x not in keys:
                keys.append(x)
            columns = {}
            for key in keys:
                try:
                    columns[key] = [row[key] for row in rows]
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"Every row must provide column {key!r}") from exc
        else:
            if x is not None:
                raise ValueError("A named x column requires a mapping or row records")
            if y is not None and len(y) != 1:
                raise ValueError("A numeric sequence represents exactly one curve")
            columns = {(y[0] if y is not None else "value"): rows}

    names = list(y) if y is not None else [name for name in columns if name != x]
    if not names:
        raise ValueError("At least one y column is required")
    curves = {}
    for name in names:
        if name not in columns:
            raise ValueError(f"Missing y column {name!r}")
        values = _values(columns[name], f"column {name!r}")
        if not values:
            raise ValueError(f"Column {name!r} must not be empty")
        try:
            curves[name] = [float('nan') if v is None else float(v.item() if callable(getattr(v, 'item', None)) else v)
                            for v in values]
        except (ValueError, TypeError, RuntimeError) as exc:
            raise ValueError(f"Y column {name!r} must contain numeric scalars or None") from exc
    length = len(next(iter(curves.values())))
    if any(len(values) != length for values in curves.values()):
        raise ValueError("All y columns must have the same length")
    if x is None:
        x_values = list(range(length))
    else:
        if x not in columns:
            raise ValueError(f"Missing x column {x!r}")
        x_values = _values(columns[x], f"column {x!r}")
        if len(x_values) != length:
            raise ValueError("X and y columns must have the same length")
    return x_values, curves


def line_plot(
    output_path: Union[str, Path],
    *,
    x: Optional[str] = None,
    y: Optional[Union[str, Sequence[str]]] = None,
    data_getter: Optional[Callable[[Any], Any]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    labels: Optional[Mapping[str, str]] = None,
    marker: Optional[str] = None,
    skip_missing: bool = False,
    figsize: tuple = (8, 5),
    dpi: int = 150,
    show: bool = False,
) -> Callable:
    """Plot a synchronous function's returned data, then return it unchanged.

    ``output_path`` specifies the figure filename (e.g. PNG, PDF or SVG). Parent
    folders are created as needed; repeated calls replace the same figure.
    Plotting happens after successful function completion, not live updates.

    Supported data:
      * Column mapping: ``{'time': [...], 'temperature': [...], ...}``.
      * Row records: ``[{'time': 0, 'temperature': 20}, ...]``.
      * One numeric sequence: ``[3, 2, 1]``; x defaults to zero-based indices.

    Use ``x``/``y`` for column names. Without ``y``, all columns except ``x``
    become curves. ``data_getter(result)`` selects nested history/tuple elements
    or converts custom results to one of the supported formats. It never
    replaces the function's original returned object. NumPy/tensor columns are
    copied for plotting; None/NaN y values create gaps by default. Set
    skip_missing=True to connect each curve's finite observations independently.
    Named x values can be
    numbers, dates or categories supported by Matplotlib.

    Matplotlib is imported lazily. No backend/global style is changed. For a
    headless server use MPLBACKEND=Agg; figures are closed even on save errors.

    Example::

        @line_plot('plots/weather.png', x='hour', y=['temperature', 'humidity'])
        def measure():
            return {'hour': [1, 2, 3], 'temperature': [20, 21, 19],
                    'humidity': [60, 58, 65]}

        @line_plot('plots/loss.png', x='epoch', y=['loss', 'val_loss'],
                   data_getter=lambda result: result['history'])
        def train():
            # Accumulate ordinary lists during training, then return them.
            return {'history': {'epoch': [1, 2], 'loss': [0.8, 0.6],
                                'val_loss': [0.9, 0.7]}, 'model': ...}
    """
    path = Path(output_path)
    if not path.suffix:
        raise ValueError("output_path must include a figure extension such as .png")
    names = None if y is None else ([y] if isinstance(y, str) else list(y))
    if names is not None and (not names or any(not isinstance(n, str) for n in names)
                              or len(set(names)) != len(names)):
        raise ValueError("y must be a column name or a nonempty sequence of unique names")
    if x is not None and not isinstance(x, str):
        raise TypeError("x must be a column name or None")
    if data_getter is not None and not callable(data_getter):
        raise TypeError("data_getter must be callable")
    if len(figsize) != 2 or min(figsize) <= 0 or dpi <= 0:
        raise ValueError("figsize must have two positive dimensions and dpi must be positive")

    def decorate(function: Callable) -> Callable:
        if inspect.iscoroutinefunction(function) or inspect.isgeneratorfunction(function):
            raise TypeError("line_plot expects a synchronous function returning collected data")

        @wraps(function)
        def wrapped(*args, **kwargs):
            result = function(*args, **kwargs)
            data = result if data_getter is None else data_getter(result)
            x_values, curves = _plot_data(data, x, names)
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(figsize=figsize)
            try:
                for name, values in curves.items():
                    curve_x, curve_y = x_values, values
                    if skip_missing:
                        pairs = [(x_value, value) for x_value, value in zip(x_values, values) if math.isfinite(value)]
                        if not pairs:
                            continue
                        curve_x, curve_y = zip(*pairs)
                    axes.plot(curve_x, curve_y, label=(labels or {}).get(name, str(name)), marker=marker)
                axes.set_title(title if title is not None else function.__name__)
                axes.set_xlabel(xlabel if xlabel is not None else (x if x is not None else 'Index'))
                axes.set_ylabel(ylabel if ylabel is not None else 'Value')
                axes.grid(True, alpha=0.3)
                axes.legend()
                figure.tight_layout()
                path.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(path, dpi=dpi)
                if show:
                    plt.show()
            finally:
                plt.close(figure)
            return result

        return wrapped

    return decorate
