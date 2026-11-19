"""Calendar-month pricing contract used only by local development authorities."""
from calendar import monthrange
from datetime import datetime, timezone

from .live_budget import _fields, _money, _require, _timestamp

SOURCE = 'https://nebius.com/services/token-factory/models/qwen-models-inference'


def month_expiry(confirmed: datetime) -> datetime:
    """Same UTC clock time next month, clamped to that month's last day."""
    _require(isinstance(confirmed, datetime) and confirmed.utcoffset() is not None)
    start = confirmed.astimezone(timezone.utc)
    year, month = (start.year + 1, 1) if start.month == 12 else (start.year, start.month + 1)
    _require(year <= 9999)
    return start.replace(year=year, month=month, day=min(start.day, monthrange(year, month)[1]))


def validate_price(price, now=None):
    _fields(price, 'input_per_million_usd output_per_million_usd source confirmed_at valid_until')
    _require(price['source'] == SOURCE)
    _require(all(0 < _money(price[k]) <= 1000 for k in ('input_per_million_usd', 'output_per_million_usd')))
    start, end = _timestamp(price['confirmed_at']), _timestamp(price['valid_until'])
    _require(start.utcoffset() is not None and end.utcoffset() is not None
             and end == month_expiry(start))
    if now is not None:
        _require(now.utcoffset() is not None and start <= now < end)
