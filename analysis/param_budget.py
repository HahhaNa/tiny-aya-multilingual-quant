"""從 config.json 逐項推導參數量與各 arm 的權重大小。面試白板題 #1 的算式來源。"""
V, H, L, KV, I, HEADS = 262144, 2048, 36, 4, 11008, 16
hd = H // HEADS                                   # head_dim = 128
emb  = V * H
attn = H*H + H*(KV*hd)*2 + H*H                    # q, k, v, o (GQA: kv 只有 4 head)
mlp  = 3 * H * I                                  # gate, up, down
per_layer = attn + mlp
body = L * per_layer
total = emb + body                                # tie_word_embeddings=True → lm_head 不另計

print(f"embedding      {emb/1e6:8.1f} M   ({V} x {H})")
print(f"每層 attn      {attn/1e6:8.1f} M   (GQA {HEADS}q/{KV}kv, head_dim {hd})")
print(f"每層 MLP       {mlp/1e6:8.1f} M   (3 x {H} x {I})")
print(f"每層合計       {per_layer/1e6:8.1f} M")
print(f"body ({L}層)   {body/1e9:8.3f} B")
print(f"總參數         {total/1e9:8.3f} B")
print(f"embedding 佔比 {emb/total*100:8.1f} %   <-- H2 的立足點")
print()
print(f"{'arm':16}{'bits/param':>12}{'GiB':>8}{'GB':>8}{'vs C':>8}")
GB = 1024**3
def size(bits_body, bits_emb, g=64):
    eff = lambda b: b + 32/g                      # scale+bias 各 16-bit 攤在 g 個權重上
    return (body*eff(bits_body) + emb*eff(bits_emb)) / 8 / GB
rows = [("A-bf16", (body+emb)*16/8/GB),   # bf16 不分 group，無 scale/bias 開銷
         ("B-q8-g64", size(8,8)), ("C-q4-g64", size(4,4)),
        ("D-q4-g32", (body*(4+32/32) + emb*(4+32/32))/8/GB), ("E-q4-emb8", size(4,8))]
c = rows[2][1]
for n, s in rows:
    print(f"{n:16}{s*8*GB/total:12.2f}{s:8.2f}{s*GB/1e9:8.2f}{s/c:7.2f}x")
