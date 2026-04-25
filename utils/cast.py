# Copyright 2026 kinorax

def str_or_none(v):
    return None if v is None else str(v)

def int_or_none(v):
    return None if v is None else int(v)

def float_or_none(v):
    return None if v is None else float(v)

# v3 では inputs と outputs の id が重複できないため、output 側の id は input 側 id を大文字にして作る。
def out_id(input_id: str) -> str:
    return str(input_id).upper()
