import struct, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
raw = (ROOT/'output/goldens/node_conv_200.goldout').read_bytes()
_, nv, _, spv, bps = struct.unpack('<4sIIII', raw[:20])
gold = np.frombuffer(raw[20:20+spv*bps], dtype=np.int8).astype(np.int32).reshape(spv, bps)  # frame0 [3136,64]
lines = (ROOT/'output/conv200_realvalue_out.hex').read_text().split()
xbeats = [i for i, L in enumerate(lines) if 'x' in L.lower() or 'z' in L.lower()]
print('total beats', len(lines), 'beats containing X/Z:', len(xbeats), 'first:', xbeats[:8])
# each line = 256-bit beat hex (64 hex chars), MSB-first. byte k (k=0..31) low->high:
# data_out = out_pix[out_idx*256 +: 256]; channel ch of tile = byte ch. Beat0=ch0..31, beat1=ch32..63.
beats = np.zeros((len(lines), 32), dtype=np.int32)
for li, L in enumerate(lines):
    L = L.zfill(64)
    for i in range(32):  # hex char pos: byte31 is leftmost
        hh = L[(31-i)*2:(31-i)*2+2]
        if 'x' in hh.lower() or 'z' in hh.lower():
            beats[li, i] = -999  # mark unknown
        else:
            beats[li, i] = int(np.int8(np.uint8(int(hh, 16))))
pix = beats.reshape(spv, 2, 32).reshape(spv, 64)
known = pix != -999
d = np.abs(pix - gold)
print('gold', gold.shape, 'rtl', pix.shape, 'known-elems', int(known.sum()), '/', pix.size)
dk = d[known]
print('mismatch(known)', int((dk != 0).sum()), '/', int(known.sum()), 'maxerr', int(dk.max()) if dk.size else 0)
print('rtl  range[%d,%d] mean=%.3f zero=%d' % (pix.min(), pix.max(), pix.mean(), (pix == 0).sum()))
print('gold range[%d,%d] mean=%.3f zero=%d' % (gold.min(), gold.max(), gold.mean(), (gold == 0).sum()))
# also try beat-order swapped (in case low/high tile order differs)
pix2 = beats.reshape(spv, 2, 32)[:, ::-1, :].reshape(spv, 64)
k2 = pix2 != -999
d2 = np.abs(pix2 - gold)[k2]
print('[swap-beat] mismatch(known)', int((d2 != 0).sum()), 'maxerr', int(d2.max()) if d2.size else 0)
# restrict to pixels with NO unknown channels, exclude image-border (top/bottom/left/right) rows
full = known.all(axis=1)
print('pixels fully-known:', int(full.sum()), '/', spv)
dfull = np.abs(pix[full] - gold[full])
print('fully-known pixels: mismatch', int((dfull != 0).sum()), '/', dfull.size, 'maxerr', int(dfull.max()) if dfull.size else 0)
print('  rtl(fk)  range[%d,%d] mean=%.3f' % (pix[full].min(), pix[full].max(), pix[full].mean()))
print('  gold(fk) range[%d,%d] mean=%.3f' % (gold[full].min(), gold[full].max(), gold[full].mean()))
